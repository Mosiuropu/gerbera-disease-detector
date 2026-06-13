"""
Gerbera Disease Classifier - Prediction / Inference
=====================================================
Load a trained model and predict disease from a flower image.
Can be used standalone or imported by the Gradio app.

Usage:
    python src/predict.py --image path/to/flower.jpg --model models/best_model.pth
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from train import build_model
from disease_db import (
    DISEASE_CLASSES,
    DISEASE_DB,
    get_disease_info,
    format_treatment_advice,
)
from preprocess import get_val_transforms


class GerberaPredictor:
    """Load a trained model and predict gerbera diseases."""

    def __init__(
        self,
        model_path: str,
        class_names_path: str = None,
        model_name: str = "efficientnet_b0",
        device: str = "auto",
        img_size: int = 224,
    ):
        # Device selection
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Load class names
        if class_names_path and Path(class_names_path).exists():
            with open(class_names_path) as f:
                self.class_names = json.load(f)
        else:
            self.class_names = DISEASE_CLASSES

        self.num_classes = len(self.class_names)

        # Build and load model
        self.model = build_model(
            num_classes=self.num_classes,
            model_name=model_name,
            pretrained=False,
        )

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Transforms
        self.transform = get_val_transforms(img_size)

        print(f"✅ Model loaded: {model_path}")
        print(f"   Device: {self.device}")
        print(f"   Classes: {self.num_classes}")

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> list:
        """
        Predict disease from a PIL image.
        
        Args:
            image: PIL Image
            top_k: Number of top predictions to return
            
        Returns:
            List of dicts: [{"class": ..., "confidence": ..., "disease_info": ...}, ...]
        """
        # Ensure top_k does not exceed number of classes
        top_k = min(top_k, self.num_classes)

        # Preprocess
        img_np = np.array(image)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.shape[2] == 4:  # RGBA
            img_np = img_np[:, :, :3]

        augmented = self.transform(image=img_np)
        tensor = augmented["image"].unsqueeze(0).to(self.device)

        # Predict
        outputs = self.model(tensor)
        probs = F.softmax(outputs, dim=1)[0]

        # Get top-k predictions
        top_probs, top_indices = probs.topk(top_k)

        results = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = self.class_names[idx.item()]
            confidence = prob.item()

            disease_info = get_disease_info(class_name)
            results.append({
                "class": class_name,
                "confidence": confidence,
                "display_name": disease_info["name"],
                "disease_info": disease_info,
            })

        return results

    def predict_and_format(self, image: Image.Image) -> str:
        """Predict and return formatted treatment advice."""
        results = self.predict(image, top_k=1)
        top = results[0]
        return format_treatment_advice(top["class"], top["confidence"])

    def predict_with_details(self, image: Image.Image) -> dict:
        """Return detailed prediction results."""
        results = self.predict(image, top_k=3)
        return {
            "top_prediction": results[0],
            "alternatives": results[1:],
            "all_predictions": [
                {"class": r["class"], "confidence": r["confidence"]}
                for r in results
            ],
        }


def main():
    parser = argparse.ArgumentParser(description="Gerbera Disease Predictor")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--model", type=str, default="models/best_model.pth")
    parser.add_argument("--class_names", type=str, default="results/class_names.json")
    parser.add_argument("--model_arch", type=str, default="efficientnet_b0")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    # Load model
    predictor = GerberaPredictor(
        model_path=args.model,
        class_names_path=args.class_names,
        model_name=args.model_arch,
        device=args.device,
    )

    # Load and predict
    image = Image.open(args.image).convert("RGB")
    results = predictor.predict(image, top_k=args.top_k)

    print(f"\n🌻 Prediction for: {args.image}")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        info = r["disease_info"]
        print(f"\n  #{i} {info['icon']} {r['display_name']}")
        print(f"     Confidence: {r['confidence']:.1%}")
        print(f"     Type: {info['type']}")
        print(f"     Pathogen: {info['pathogen']}")

    # Show full treatment for top prediction
    print("\n" + "=" * 60)
    print("DETAILED TREATMENT RECOMMENDATION:")
    print("=" * 60)
    print(predictor.predict_and_format(image))


if __name__ == "__main__":
    main()
