"""Visualize train/val Dice and loss across epochs.

Usage:
    python visualize.py --metrics_path data/preprocessed_data/metrics.csv --out plot.png

The metrics file is produced by train.py (JSON or CSV with columns epoch, train_loss, val_loss, train_dice, val_dice).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import csv


def load_metrics(path: Path):
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        return data
    elif path.suffix == ".csv":
        with open(path) as f:
            reader = csv.DictReader(f)
            return [
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_dice": float(row["train_dice"]),
                    "val_loss": float(row["val_loss"]),
                    "val_dice": float(row["val_dice"]),
                }
                for row in reader
            ]
    else:
        raise ValueError(f"Unsupported metrics format: {path.suffix}")


def plot_dice(metrics, out_path: Path | None = None):
    epochs = [m["epoch"] for m in metrics]
    train_dice = [m["train_dice"] for m in metrics]
    val_dice = [m["val_dice"] for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss subplot
    train_loss = [m["train_loss"] for m in metrics]
    val_loss = [m["val_loss"] for m in metrics]
    axes[0].plot(epochs, train_loss, marker="o", label="Train loss")
    axes[0].plot(epochs, val_loss, marker="s", label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Dice subplot
    axes[1].plot(epochs, train_dice, marker="o", label="Train Dice")
    axes[1].plot(epochs, val_dice, marker="s", label="Val Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice score")
    axes[1].set_title("Dice vs Epoch")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot Dice curves from metrics file")
    parser.add_argument("--metrics_path", type=str, required=True,
                        help="Path to metrics.json or metrics.csv produced by train.py")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional path to save the plot (e.g., plot.png). If omitted, shows interactively.")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    metrics = load_metrics(metrics_path)
    if len(metrics) == 0:
        raise RuntimeError("Metrics file is empty—run training first.")

    out_path = Path(args.out) if args.out else None
    plot_dice(metrics, out_path)


if __name__ == "__main__":
    main()