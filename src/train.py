"""
Gerbera Disease Classifier - Training Script
=============================================
Transfer learning with EfficientNet-B0 for gerbera disease classification.
Optimized for free GPU training on Kaggle/Colab.

Usage (local):
    python src/train.py --data_dir data/split --epochs 30 --batch_size 32

Usage (Kaggle/Colab):
    !python src/train.py --data_dir /kaggle/input/gerbera-dataset --epochs 25
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from preprocess import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Custom Dataset with Albumentations
# ---------------------------------------------------------------------------
class TransformedDataset(datasets.ImageFolder):
    """ImageFolder with albumentations transforms."""

    def __init__(self, root, transform=None, **kwargs):
        super().__init__(root, **kwargs)
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)

        # Convert PIL to numpy for albumentations
        image = np.array(image)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------
def build_model(
    num_classes: int,
    model_name: str = "efficientnet_b0",
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Build a transfer learning model.
    
    Args:
        num_classes: Number of disease classes
        model_name: Architecture name (efficientnet_b0, resnet50, mobilenet_v3)
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate for regularization
    
    Returns:
        PyTorch model ready for fine-tuning
    """
    if model_name == "efficientnet_b0":
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = models.efficientnet_b0(weights=weights)

        # Replace classifier head
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
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None
        model = models.resnet50(weights=weights)

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
        if pretrained:
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = models.mobilenet_v3_large(weights=weights)

        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
class Trainer:
    """Handles the full training pipeline."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        class_names: list,
        device: str = "auto",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
    ):
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        print(f"🖥️  Device: {self.device}")
        if self.device.type == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.class_names = class_names

        # Loss with label smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=learning_rate * 0.01,
        )

        # Tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1": [],
            "learning_rate": [],
        }
        self.best_val_f1 = 0.0
        self.best_model_state = None
        self.patience_counter = 0

    def train_epoch(self) -> tuple:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self) -> tuple:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Macro F1 for imbalanced classes
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return epoch_loss, epoch_acc, f1, all_preds, all_labels

    def train(
        self,
        num_epochs: int = 30,
        patience: int = 10,
        save_dir: str = "models",
    ):
        """Full training loop with early stopping."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🚀 Starting training for {num_epochs} epochs")
        print(f"   Early stopping patience: {patience}")
        print(f"   Classes: {self.num_classes}")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")
        print("-" * 60)

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, val_f1, _, _ = self.validate()

            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["learning_rate"].append(current_lr)

            epoch_time = time.time() - epoch_start

            # Print progress
            improved = ""
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                torch.save(self.best_model_state, save_dir / "best_model.pth")
                self.patience_counter = 0
                improved = " ★ BEST"
            else:
                self.patience_counter += 1

            print(
                f"  Epoch {epoch:3d}/{num_epochs} | "
                f"Train: {train_loss:.4f}/{train_acc:.1%} | "
                f"Val: {val_loss:.4f}/{val_acc:.1%}/{val_f1:.3f} | "
                f"LR: {current_lr:.1e} | "
                f"{epoch_time:.1f}s{improved}"
            )

            # Save last model
            torch.save(
                {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                save_dir / "last_model.pth",
            )

            # Early stopping
            if self.patience_counter >= patience:
                print(f"\n⚠️  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"✅ Training complete in {total_time / 60:.1f} minutes")
        print(f"   Best validation F1: {self.best_val_f1:.3f}")

        # Load best model for final evaluation
        if self.best_model_state:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in self.best_model_state.items()}
            )

        return self.history

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Final evaluation with detailed metrics."""
        self.model.eval()
        all_preds = []
        all_labels = []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            outputs = self.model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Classification report
        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.class_names,
            digits=3,
            zero_division=0,
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Overall metrics
        metrics = {
            "accuracy": float(np.mean(all_preds == all_labels)),
            "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
            "macro_precision": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

        print("\n📊 Final Evaluation Metrics:")
        print(f"   Accuracy:    {metrics['accuracy']:.1%}")
        print(f"   Macro F1:    {metrics['macro_f1']:.3f}")
        print(f"   Weighted F1: {metrics['weighted_f1']:.3f}")
        print(f"\n{report}")

        return metrics


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def create_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 2,
) -> tuple:
    """Create train/val/test dataloaders."""
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    # Use val dir as test if test doesn't exist
    if not test_dir.exists():
        test_dir = val_dir

    train_dataset = TransformedDataset(
        train_dir, transform=get_train_transforms(img_size)
    )
    val_dataset = TransformedDataset(
        val_dir, transform=get_val_transforms(img_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_names = train_dataset.classes
    print(f"\n📁 Dataset loaded:")
    print(f"   Train: {len(train_dataset)} images ({len(class_names)} classes)")
    print(f"   Val:   {len(val_dataset)} images")
    print(f"   Classes: {class_names}")

    return train_loader, val_loader, class_names


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_history(history: dict, save_dir: str = "results"):
    """Plot training curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        epochs = range(1, len(history["train_loss"]) + 1)

        # Loss
        axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
        axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc")
        axes[1].plot(epochs, history["val_acc"], "r-", label="Val Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training & Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # F1 Score
        axes[2].plot(epochs, history["val_f1"], "g-", label="Val Macro F1")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("F1 Score")
        axes[2].set_title("Validation Macro F1")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n📈 Training curves saved to {save_dir / 'training_curves.png'}")

    except ImportError:
        print("matplotlib not available, skipping plots")


def plot_confusion_matrix(cm, class_names, save_dir="results"):
    """Plot confusion matrix heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            title="Confusion Matrix",
            ylabel="True Label",
            xlabel="Predicted Label",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        thresh = max(cm) / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(
                    j, i, format(cm[i][j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black",
                    fontsize=8,
                )

        plt.tight_layout()
        plt.savefig(save_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📊 Confusion matrix saved to {save_dir / 'confusion_matrix.png'}")

    except ImportError:
        print("matplotlib not available, skipping confusion matrix plot")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Gerbera Disease Classifier")
    parser.add_argument("--data_dir", type=str, default="data/split", help="Path to split dataset")
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet50", "mobilenet_v3"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Run scraper.py first, then preprocess.py to split the dataset.")
        return

    print("\n🌻 GERBERA DISEASE CLASSIFIER - TRAINING")
    print("=" * 60)
    print(f"   Model: {args.model}")
    print(f"   Data:  {data_dir}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")

    # Load data
    train_loader, val_loader, class_names = create_dataloaders(
        data_dir, args.batch_size, args.img_size
    )

    # Build model
    model = build_model(
        num_classes=len(class_names),
        model_name=args.model,
        pretrained=True,
        dropout=args.dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔧 Model: {total_params:,} total params, {trainable_params:,} trainable")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=len(class_names),
        class_names=class_names,
        learning_rate=args.lr,
        device=args.device,
    )

    history = trainer.train(
        num_epochs=args.epochs,
        patience=args.patience,
        save_dir=args.save_dir,
    )

    # Evaluate
    metrics = trainer.evaluate()

    # Save artifacts
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(
            {k: v for k, v in metrics.items() if k != "classification_report"},
            f, indent=2,
        )

    with open(results_dir / "classification_report.txt", "w") as f:
        f.write(metrics["classification_report"])

    # Save class names
    with open(results_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    # Plot
    plot_history(history, args.results_dir)
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        class_names,
        args.results_dir,
    )

    print(f"\n✅ All artifacts saved to:")
    print(f"   Model:      {args.save_dir}/best_model.pth")
    print(f"   Metrics:    {results_dir}/metrics.json")
    print(f"   Plots:      {results_dir}/training_curves.png")
    print(f"   Classes:    {results_dir}/class_names.json")
    print(f"\n   Best val F1: {trainer.best_val_f1:.3f}")


if __name__ == "__main__":
    main()
