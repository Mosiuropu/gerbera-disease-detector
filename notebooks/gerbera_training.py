"""
Gerbera Disease Classifier - Kaggle/Colab Training Notebook
============================================================
Copy this to a Kaggle Notebook or Google Colab for free GPU training.

Steps:
1. Upload your dataset to Kaggle/Colab
2. Enable GPU: Settings → Accelerator → GPU
3. Run all cells

Dataset structure expected:
    /kaggle/input/gerbera-dataset/
        train/
            healthy/
            powdery_mildew/
            botrytis_blight/
            ...
        val/
            healthy/
            ...
"""

# ============================================================
# Cell 1: Setup & Installation
# ============================================================
# !pip install timm albumentations -q

import os
import sys
import json
import time
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models
from torch.optim.lr_scheduler import CosineAnnealingLR

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm

# ============================================================
# Cell 2: Configuration
# ============================================================
# ----- MODIFY THESE PATHS FOR YOUR ENVIRONMENT -----
# Kaggle:     "/kaggle/input/gerbera-dataset/"
# Colab:      "/content/drive/MyDrive/gerbera-dataset/"
# Local:      "data/split/"

DATA_DIR = Path("/kaggle/input/gerbera-dataset")  # ← Change this!
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIG = {
    "model_name": "efficientnet_b0",  # Options: efficientnet_b0, resnet50, mobilenet_v3
    "img_size": 224,
    "batch_size": 32,
    "num_epochs": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "label_smoothing": 0.1,
    "patience": 10,
    "num_workers": 2,
    "seed": 42,
}

# Set seeds for reproducibility
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ============================================================
# Cell 3: Data Exploration
# ============================================================
print("\n📊 Dataset Statistics:")
print("=" * 50)

class_counts = {}
for split in ["train", "val", "test"]:
    split_dir = DATA_DIR / split
    if split_dir.exists():
        print(f"\n{split}/")
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                count = len(list(cls_dir.glob("*.*")))
                class_counts[cls_dir.name] = class_counts.get(cls_dir.name, 0) + count
                bar = "█" * (count // 5) if count > 0 else ""
                print(f"  {cls_dir.name:25s} {count:5d}  {bar}")

print(f"\nTotal images: {sum(class_counts.values())}")
print(f"Classes: {len(class_counts)}")

# Class names
if (DATA_DIR / "train").exists():
    CLASS_NAMES = sorted([d.name for d in (DATA_DIR / "train").iterdir() if d.is_dir()])
else:
    CLASS_NAMES = sorted(class_counts.keys())
NUM_CLASSES = len(CLASS_NAMES)
print(f"Class names: {CLASS_NAMES}")

# ============================================================
# Cell 4: Augmentation Pipelines
# ============================================================
def get_train_transforms(img_size=224):
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=img_size//8, max_width=img_size//8, fill_value=0, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ============================================================
# Cell 5: Custom Dataset
# ============================================================
class GerberaDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        root = Path(root)
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name
            cls_idx = CLASS_NAMES.index(cls_name) if cls_name in CLASS_NAMES else -1
            if cls_idx < 0:
                continue
            for img_path in cls_dir.glob("*.*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    self.samples.append((str(img_path), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

# ============================================================
# Cell 6: Create DataLoaders
# ============================================================
train_dataset = GerberaDataset(DATA_DIR / "train", transform=get_train_transforms(CONFIG["img_size"]))
val_dataset = GerberaDataset(DATA_DIR / "val", transform=get_val_transforms(CONFIG["img_size"]))

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=CONFIG["num_workers"],
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=True,
)

print(f"\n📁 DataLoaders ready:")
print(f"   Train: {len(train_dataset)} images, {len(train_loader)} batches")
print(f"   Val:   {len(val_dataset)} images, {len(val_loader)} batches")

# ============================================================
# Cell 7: Build Model
# ============================================================
def build_model(num_classes, model_name="efficientnet_b0", dropout=0.3):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )
    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

model = build_model(NUM_CLASSES, CONFIG["model_name"], CONFIG["dropout"])
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"🔧 Model: {CONFIG['model_name']}")
print(f"   Total params: {total_params:,}")
print(f"   Trainable params: {trainable_params:,}")

# ============================================================
# Cell 8: Training Loop
# ============================================================
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"], eta_min=CONFIG["learning_rate"] * 0.01)

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
best_f1 = 0.0
best_model_state = None
patience_counter = 0

print(f"\n🚀 Training for {CONFIG['num_epochs']} epochs...")
print("=" * 80)

for epoch in range(1, CONFIG["num_epochs"] + 1):
    epoch_start = time.time()

    # --- Train ---
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    train_loss /= train_total
    train_acc = train_correct / train_total

    # --- Validate ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= val_total
    val_acc = val_correct / val_total
    val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    epoch_time = time.time() - epoch_start

    improved = ""
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
        improved = " ★ BEST"
    else:
        patience_counter += 1

    print(
        f"  Epoch {epoch:3d}/{CONFIG['num_epochs']} | "
        f"Train: {train_loss:.4f}/{train_acc:.1%} | "
        f"Val: {val_loss:.4f}/{val_acc:.1%}/{val_f1:.3f} | "
        f"{epoch_time:.1f}s{improved}"
    )

    if patience_counter >= CONFIG["patience"]:
        print(f"\n⚠️  Early stopping at epoch {epoch}")
        break

# ============================================================
# Cell 9: Save Model
# ============================================================
if best_model_state:
    torch.save(best_model_state, OUTPUT_DIR / "best_model.pth")

    # Save class names
    with open(OUTPUT_DIR / "class_names.json", "w") as f:
        json.dump(CLASS_NAMES, f, indent=2)

    # Save config
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Save history
    with open(OUTPUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Model saved to {OUTPUT_DIR}")

# ============================================================
# Cell 10: Final Evaluation
# ============================================================
model.load_state_dict(best_model_state)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n📊 Final Classification Report:")
print("=" * 60)
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=3, zero_division=0)
print(report)

with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
    f.write(report)

# ============================================================
# Cell 11: Training Curves
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
epochs = range(1, len(history["train_loss"]) + 1)

axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
axes[0].set_title("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, history["train_acc"], "b-", label="Train")
axes[1].plot(epochs, history["val_acc"], "r-", label="Val")
axes[1].set_title("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs, history["val_f1"], "g-", label="Val Macro F1")
axes[2].set_title("Macro F1 Score")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"📈 Training curves saved")

# ============================================================
# Cell 12: Confusion Matrix
# ============================================================
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(NUM_CLASSES),
    yticks=np.arange(NUM_CLASSES),
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    title="Confusion Matrix",
    ylabel="True",
    xlabel="Predicted",
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

thresh = cm.max() / 2.0
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, format(cm[i][j], "d"), ha="center", va="center",
                color="white" if cm[i][j] > thresh else "black", fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\n✅ All outputs saved to {OUTPUT_DIR}/")
print(f"   - best_model.pth")
print(f"   - class_names.json")
print(f"   - config.json")
print(f"   - history.json")
print(f"   - classification_report.txt")
print(f"   - training_curves.png")
print(f"   - confusion_matrix.png")
