"""
# ###############################################################
# Choose super_class Dice or general Dice to check best model score !
# ###############################################################
"""
import csv
from pathlib import Path

import torch


MODEL_PATH = "data/processed/best_model.pt"
CSV_PATH = "metrics/metrics_flat_super_dice.csv"


def print_ckpt_metadata(path: str = MODEL_PATH):
    try:
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "epoch" in checkpoint:
            print("---------------------------------------")
            print(f"Model saved at Epoch:            {checkpoint['epoch']}")
            print(f"Best Validation Loss:           {checkpoint['val_loss']:.4f}")
            print(f"Best Validation Dice:           {checkpoint['val_dice']:.4f}")
            if "val_super_dice" in checkpoint:
                print(f"Best Validation Superclass Dice:{checkpoint['val_super_dice']:.4f}")
            print("---------------------------------------")
        else:
            print("This file contains only the model weights, no score data.")
    except FileNotFoundError:
        print(f"Could not find checkpoint at {path}.")
    except Exception as e:
        print(f"Error reading file: {e}")


def print_csv_bests(path: str = CSV_PATH):
    p = Path(path)
    if not p.exists():
        print(f"Metrics CSV not found at {p}.")
        return
    try:
        with open(p) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            print(f"Metrics CSV {p} is empty.")
            return

        # find max val_super_dice and val_dice
        best_super = max(
            (r for r in rows if r.get("val_super_dice") not in ("", None, "nan")),
            key=lambda r: float(r["val_super_dice"]),
            default=None,
        )
        best_dice = max(rows, key=lambda r: float(r["val_dice"]))

        print("CSV best metrics (validation):")
        if best_super:
            print(
                f"  Best val_super_dice: {float(best_super['val_super_dice']):.4f}"
                f" @ epoch {best_super['epoch']}, val_dice={float(best_super['val_dice']):.4f}"
            )
        else:
            print("  No val_super_dice found in CSV.")

        print(
            f"  Best val_dice:        {float(best_dice['val_dice']):.4f}"
            f" @ epoch {best_dice['epoch']}"
        )
        print("---------------------------------------")
    except Exception as e:
        print(f"Error reading CSV {p}: {e}")


if __name__ == "__main__":
    print_ckpt_metadata(MODEL_PATH)
    print_csv_bests(CSV_PATH)

