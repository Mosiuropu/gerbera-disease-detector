"""
Image Preprocessing & Augmentation
===================================
Handles dataset preparation, splitting, and augmentation for training.
Compatible with both local execution and Kaggle/Colab notebooks.
"""

import os
import shutil
import random
from pathlib import Path
from collections import Counter

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(img_size: int = 224) -> A.Compose:
    """Training augmentations - aggressive to expand small datasets."""
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.4,
        ),
        A.Affine(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=30,
            p=0.5,
        ),
        A.OneOf([
            A.GaussNoise(),
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.1),
            A.GridDistortion(num_steps=5, distort_limit=0.1),
        ], p=0.2),
        A.CoarseDropout(p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 224) -> A.Compose:
    """Validation/test augmentations - only resize + normalize."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def verify_images(data_dir: Path) -> dict:
    """Verify and count images in each class directory."""
    stats = {}
    corrupted = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        count = 0
        for img_path in class_dir.glob("*.*"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                continue
            try:
                img = cv2.imread(str(img_path))
                if img is None or img.size == 0:
                    corrupted.append(str(img_path))
                    continue
                count += 1
            except Exception:
                corrupted.append(str(img_path))

        stats[class_dir.name] = count

    if corrupted:
        print(f"\n⚠️  Found {len(corrupted)} corrupted images (will be skipped)")

    return stats


def split_dataset(
    data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Split raw dataset into train/val/test sets.
    
    Expected input structure:
        data_dir/
            healthy/
                img1.jpg
                img2.jpg
            powdery_mildew/
                img3.jpg
                ...
    
    Output structure:
        output_dir/
            train/
                healthy/
                powdery_mildew/
            val/
                healthy/
                powdery_mildew/
            test/
                healthy/
                powdery_mildew/
    """
    random.seed(seed)

    print("\n📂 Splitting dataset into train/val/test...")
    print(f"   Ratios: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")

    stats = verify_images(data_dir)
    total = sum(stats.values())
    print(f"\n   Found {total} images across {len(stats)} classes:")
    for cls, count in sorted(stats.items()):
        print(f"     {cls}: {count}")

    # Create output directories
    for split in ["train", "val", "test"]:
        for cls in stats.keys():
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    # Split each class
    for cls, count in stats.items():
        cls_dir = data_dir / cls
        images = [
            f for f in cls_dir.glob("*.*")
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        ]
        random.shuffle(images)

        n_train = int(len(images) * train_ratio)
        n_val = int(len(images) * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for img in train_imgs:
            shutil.copy2(img, output_dir / "train" / cls / img.name)
        for img in val_imgs:
            shutil.copy2(img, output_dir / "val" / cls / img.name)
        for img in test_imgs:
            shutil.copy2(img, output_dir / "test" / cls / img.name)

        print(f"   {cls}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

    print(f"\n✅ Dataset split complete → {output_dir}")


def augment_dataset(
    data_dir: Path,
    output_dir: Path,
    target_per_class: int = 500,
    img_size: int = 224,
    seed: int = 42,
):
    """
    Augment a dataset to reach a target number of images per class.
    Reads original images, applies augmentation transforms, and saves results.
    """
    random.seed(seed)
    np.random.seed(seed)

    print("\n🔄 Augmenting dataset...")
    print(f"   Target: {target_per_class} images per class")

    aug = A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.Affine(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.GaussNoise(p=0.3),
    ])

    for cls_dir in sorted(data_dir.iterdir()):
        if not cls_dir.is_dir():
            continue

        images = [
            f for f in cls_dir.glob("*.*")
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        ]

        if not images:
            continue

        out_dir = output_dir / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy originals
        for img_path in images:
            shutil.copy2(img_path, out_dir / img_path.name)

        # Generate augmented copies
        n_existing = len(images)
        n_needed = max(0, target_per_class - n_existing)

        print(f"   {cls_dir.name}: {n_existing} original → +{n_needed} augmented", end="")

        for i in range(n_needed):
            src = random.choice(images)
            img = cv2.imread(str(src))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = aug(image=img_rgb)
            aug_img = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)

            aug_name = f"aug_{cls_dir.name}_{i:04d}.jpg"
            cv2.imwrite(str(out_dir / aug_name), aug_img)

        final_count = len(list(out_dir.glob("*.*")))
        print(f" → {final_count} total")

    print("\n✅ Augmentation complete")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def create_class_weights(data_dir: Path) -> dict:
    """Calculate class weights for imbalanced datasets (inverse frequency)."""
    stats = {}
    for cls_dir in data_dir.iterdir():
        if cls_dir.is_dir():
            stats[cls_dir.name] = len(list(cls_dir.glob("*.*")))

    total = sum(stats.values())
    n_classes = len(stats)

    weights = {}
    for cls, count in stats.items():
        weights[cls] = total / (n_classes * max(count, 1))

    return weights


if __name__ == "__main__":
    # Quick test
    data_dir = Path("data/raw")
    if data_dir.exists():
        stats = verify_images(data_dir)
        print("\nDataset statistics:")
        for cls, count in sorted(stats.items()):
            print(f"  {cls}: {count}")

        weights = create_class_weights(data_dir)
        print("\nClass weights:")
        for cls, w in sorted(weights.items()):
            print(f"  {cls}: {w:.2f}")
    else:
        print("No data found at data/raw/. Run scraper.py first.")
