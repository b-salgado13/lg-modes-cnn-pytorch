"""Run a single-image prediction with a trained checkpoint.

Reproduces the logic of `PredictLGModeButtonPushed` from the original GUI
(Appendix 6): run the selected network on the chosen pattern, report the
predicted mode (medium, l, p) and the confidence percentage.

Example:
    python -m src.predict \\
        --checkpoint checkpoints/resnet18.pt \\
        --image path/to/pattern.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from src.datasets.downsampling import down_sampling
from src.datasets.image_dataset import build_transform
from src.mode_labels import parse_folder_name
from src.models.simple_cnn import SimpleCNN
from src.models.simple_mlp import SimpleMLP
from src.models.transfer_learning import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint")
    parser.add_argument("--image", required=True, help="Path to the pattern image to classify")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint["model_type"]

    if model_type == "simple_mlp":
        model = SimpleMLP(
            input_dim=checkpoint["input_dim"],
            hidden_units=checkpoint["hidden_units"],
            num_classes=checkpoint["num_classes"],
        )
    elif model_type == "simple_cnn":
        model = SimpleCNN(
            num_classes=checkpoint["num_classes"],
            in_channels=checkpoint["in_channels"],
            input_size=tuple(checkpoint["input_size"]),
            filters=checkpoint["filters"],
        )
    elif model_type == "transfer_learning":
        model = build_model(
            backbone=checkpoint["backbone"],
            num_classes=checkpoint["num_classes"],
            pretrained=False,  # weights come from the checkpoint, not ImageNet
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}' in checkpoint.")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def _predict_mlp(model, checkpoint, image_path: str, device: torch.device) -> Tuple[int, torch.Tensor]:
    x, y, width, height = checkpoint["crop_box"]
    with Image.open(image_path) as img:
        gray = img.convert("L")
        cropped = gray.crop((x, y, x + width, y + height))
        import numpy as np

        array = np.asarray(cropped, dtype="float64")
    down = down_sampling(array, checkpoint["down_square"])
    features = torch.from_numpy(down.reshape(1, -1).astype("float32")).to(device)
    with torch.no_grad():
        logits = model(features)
        probs = F.softmax(logits, dim=1)[0]
    return int(probs.argmax().item()), probs


def _predict_image_model(model, checkpoint, image_path: str, device: torch.device, channels: int, image_size) -> Tuple[int, torch.Tensor]:
    transform = build_transform(tuple(image_size), channels=channels, normalize_imagenet=(channels == 3))
    with Image.open(image_path) as img:
        tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    return int(probs.argmax().item()), probs


def predict(checkpoint_path: str, image_path: str, device: torch.device) -> Dict:
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    model_type = checkpoint["model_type"]

    if model_type == "simple_mlp":
        pred_idx, probs = _predict_mlp(model, checkpoint, image_path, device)
    elif model_type == "simple_cnn":
        pred_idx, probs = _predict_image_model(
            model, checkpoint, image_path, device, channels=1, image_size=checkpoint["input_size"]
        )
    else:  # transfer_learning
        pred_idx, probs = _predict_image_model(
            model, checkpoint, image_path, device, channels=3, image_size=checkpoint["image_size"]
        )

    idx_to_class = {idx: name for name, idx in checkpoint["class_to_idx"].items()}
    class_name = idx_to_class[pred_idx]
    mode = parse_folder_name(class_name)

    return {
        "class_name": class_name,
        "medium": mode.medium,
        "l": mode.l,
        "p": mode.p,
        "description": mode.describe(),
        "confidence": float(probs[pred_idx].item()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    result = predict(args.checkpoint, args.image, device)
    print(f"Predicted mode : {result['description']}")
    print(f"Class label    : {result['class_name']}")
    print(f"Confidence     : {result['confidence'] * 100:.2f}%")


if __name__ == "__main__":
    main()
