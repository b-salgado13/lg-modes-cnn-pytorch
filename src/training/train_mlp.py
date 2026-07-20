"""Training script for the simple MLP (Appendix 1).

Example:
    python -m src.training.train_mlp \\
        --data-dir "data/Fotos" \\
        --output checkpoints/mlp.pt \\
        --epochs 400 \\
        --hidden-units 5 \\
        --down-square 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim

from src.datasets.mlp_dataset import DEFAULT_CROP_BOX, build_mlp_dataset
from src.models.simple_mlp import SimpleMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory with m__<l> subfolders")
    parser.add_argument("--output", required=True, help="Path to save the trained checkpoint")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--hidden-units", type=int, default=5)
    parser.add_argument("--down-square", type=int, default=20, help="Block size for downsampling")
    parser.add_argument(
        "--crop-box",
        type=int,
        nargs=4,
        default=list(DEFAULT_CROP_BOX),
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Crop applied before downsampling (MATLAB imcrop convention)",
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _iterate_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool):
    n = X.shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading and preprocessing images from {args.data_dir} ...")
    dataset = build_mlp_dataset(
        args.data_dir,
        crop_box=tuple(args.crop_box),
        down_square=args.down_square,
    )
    print(f"Loaded {dataset.X.shape[0]} samples, {dataset.X.shape[1]} features, "
          f"{len(dataset.class_to_idx)} classes.")

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    X_train, X_rest, y_train, y_rest = train_test_split(
        dataset.X, dataset.y, train_size=args.train_ratio, random_state=args.seed,
        stratify=dataset.y,
    )
    val_size = args.val_ratio / (args.val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, train_size=val_size, random_state=args.seed, stratify=y_rest,
    )

    device = torch.device(args.device)
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    model = SimpleMLP(
        input_dim=dataset.X.shape[1],
        hidden_units=args.hidden_units,
        num_classes=len(dataset.class_to_idx),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in _iterate_batches(X_train_t, y_train_t, args.batch_size, shuffle=True):
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_acc = evaluate(model, X_val_t, y_val_t)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train_loss={epoch_loss / n_batches:.4f} | val_acc={val_acc * 100:.2f}%"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc = evaluate(model, X_test_t, y_test_t)
    print(f"\nBest val accuracy: {best_val_acc * 100:.2f}%")
    print(f"Test accuracy (best-val checkpoint): {test_acc * 100:.2f}%")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": dataset.X.shape[1],
            "hidden_units": args.hidden_units,
            "num_classes": len(dataset.class_to_idx),
            "class_to_idx": dataset.class_to_idx,
            "crop_box": list(args.crop_box),
            "down_square": args.down_square,
            "model_type": "simple_mlp",
        },
        output_path,
    )
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(dataset.class_to_idx, f, indent=2, ensure_ascii=False)
    print(f"Saved checkpoint to {output_path}")


if __name__ == "__main__":
    main()
