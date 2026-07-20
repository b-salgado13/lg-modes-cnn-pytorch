"""Training script for the simple 3-layer CNN (Appendix 3).

Example:
    python -m src.training.train_cnn \\
        --data-dir "data/Fotos 128x192" \\
        --output checkpoints/cnn_8_8_8.pt \\
        --epochs 20 \\
        --batch-size 32 \\
        --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from src.datasets.image_dataset import build_image_folder_dataset
from src.models.simple_cnn import SimpleCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory with one subfolder per class")
    parser.add_argument("--output", required=True, help="Path to save the trained checkpoint")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (matches 'sgdm')")
    parser.add_argument("--filters", type=int, default=8)
    parser.add_argument("--image-size", type=int, nargs=2, default=[128, 192], metavar=("H", "W"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"Building dataset from {args.data_dir} ...")
    dataset = build_image_folder_dataset(
        args.data_dir, image_size=tuple(args.image_size), channels=1
    )
    print(f"Found {len(dataset)} images across {len(dataset.classes)} classes.")

    targets = [label for _, label in dataset.samples]
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))),
        train_size=args.train_ratio,
        stratify=targets,
        random_state=args.seed,
    )
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = SimpleCNN(
        num_classes=len(dataset.classes),
        in_channels=1,
        input_size=tuple(args.image_size),
        filters=args.filters,
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_set)
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} | val_acc={val_acc * 100:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nBest validation accuracy: {best_val_acc * 100:.2f}%")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": len(dataset.classes),
            "in_channels": 1,
            "input_size": list(args.image_size),
            "filters": args.filters,
            "class_to_idx": dataset.class_to_idx,
            "model_type": "simple_cnn",
        },
        output_path,
    )
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(dataset.class_to_idx, f, indent=2, ensure_ascii=False)
    print(f"Saved checkpoint to {output_path}")


if __name__ == "__main__":
    main()
