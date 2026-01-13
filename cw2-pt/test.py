"""
Test/evaluation script for pelvic segmentation (full volumes, slice-wise 2D U-Net).

Example:
python test.py \
  --data_dir data/preprocessed_data/test \
  --flat_ckpt data/preprocessed_data/best_flat.pt \
  --hier_ckpt data/preprocessed_data/best_hier.pt \
  --num_classes 9 \
  --use_hierarchical_metrics \
  --out_csv data/preprocessed_data/test_metrics.csv \
  --out_json data/preprocessed_data/test_metrics.json

Notes:
- Expects preprocessed data layout:
    data_dir/
      images/*.nii(.gz)
      masks/*.nii(.gz)

- Uses MalePelvicDataset, which loads and transposes to (Z,Y,X) internally.
"""

import argparse
import json
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import MalePelvicDataset
from src.model import UNet
from src.losses import build_distance_matrix


# ---------- Metrics ----------

def dice_per_class(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6):
    """
    preds, targets: [Z,Y,X] int
    Returns: list length num_classes with dice (nan if class absent in both)
    """
    dices = []
    for c in range(num_classes):
        pred_c = (preds == c)
        tgt_c = (targets == c)
        denom = pred_c.sum() + tgt_c.sum()
        if denom == 0:
            dices.append(float("nan"))
            continue
        inter = (pred_c & tgt_c).sum()
        dice = (2.0 * inter.float() + eps) / (denom.float() + eps)
        dices.append(dice.item())
    return dices


def mean_foreground_dice(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6):
    """
    Mean dice across classes present in targets, ignoring background (class 0),
    matching your training-time dice.
    """
    dice_sum, count = 0.0, 0
    for c in range(1, num_classes):
        pred_c = (preds == c)
        tgt_c = (targets == c)
        denom = pred_c.sum() + tgt_c.sum()
        if denom == 0:
            continue
        inter = (pred_c & tgt_c).sum()
        dice = (2.0 * inter.float() + eps) / (denom.float() + eps)
        dice_sum += dice
        count += 1
    return (dice_sum / count).item() if count > 0 else 0.0

def binary_dice(pred_bin: torch.Tensor, tgt_bin: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (pred_bin & tgt_bin).sum()
    denom = pred_bin.sum() + tgt_bin.sum()
    if denom == 0:
        return 1.0  # both empty
    return ((2.0 * inter.float() + eps) / (denom.float() + eps)).item()


def prostate_superclass_dice(preds: torch.Tensor, targets: torch.Tensor, prostate_ids=(4, 5)) -> float:
    pred_p = torch.zeros_like(preds, dtype=torch.bool)
    tgt_p  = torch.zeros_like(targets, dtype=torch.bool)
    for pid in prostate_ids:
        pred_p |= (preds == pid)
        tgt_p  |= (targets == pid)
    return binary_dice(pred_p, tgt_p)


@torch.no_grad()
def mean_hier_cost_argmax(preds: torch.Tensor, targets: torch.Tensor, D: torch.Tensor) -> float:
    """
    preds, targets: [Z,Y,X] int
    D: [C,C] where D[true,pred]
    """
    return D[targets, preds].float().mean().item()


@torch.no_grad()
def expected_hier_cost_from_logits(logits: torch.Tensor, targets: torch.Tensor, D: torch.Tensor) -> float:
    """
    logits: [N,C,H,W]
    targets: [N,H,W]
    D: [C,C] where D[true,pred]
    returns mean expected cost over all pixels
    """
    probs = torch.softmax(logits, dim=1)              # [N,C,H,W]
    D_row = D[targets]                               # [N,H,W,C]
    exp_cost = (probs.permute(0,2,3,1) * D_row).sum(dim=-1)  # [N,H,W]
    return exp_cost.mean().item()


# ---------- Inference (slice-wise over Z) ----------

@torch.no_grad()
def predict_volume(model: torch.nn.Module, image: torch.Tensor, device: torch.device, batch_slices: int = 8):
    """
    image: [1,Z,Y,X] float tensor (from MalePelvicDataset)
    Returns:
      pred: [Z,Y,X] long
      logits_all: [Z,C,Y,X] float (optional for expected cost; returned as tensor on CPU)
    """
    model.eval()
    image = image.to(device)

    _, Z, Y, X = image.shape
    preds = torch.empty((Z, Y, X), dtype=torch.long, device="cpu")
    logits_cpu = torch.empty((Z, model.final.out_channels, Y, X), dtype=torch.float32, device="cpu")

    z = 0
    while z < Z:
        z2 = min(Z, z + batch_slices)
        # slices: [N,1,Y,X]
        x = image[:, z:z2].permute(1, 0, 2, 3).contiguous()  # [N,1,Y,X]
        logits = model(x)  # [N,C,Y,X]
        pred = torch.argmax(logits, dim=1)  # [N,Y,X]

        preds[z:z2] = pred.cpu()
        logits_cpu[z:z2] = logits.cpu()

        z = z2

    return preds, logits_cpu


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)  # supports {"model_state": ...} or raw state_dict
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------- Main ----------

def evaluate_model_on_test(model, test_loader, device, num_classes, D=None, batch_slices=8):
    """
    Returns aggregated dict + per-case list
    """
    per_case = []

    # accumulators
    mean_fg_dices = []
    per_class_dice_accum = [[] for _ in range(num_classes)]
    hcost_argmax_list = []
    hcost_expected_list = []
    prostate_dice_list = []


    for batch in test_loader:
        # MalePelvicDataset returns (image, mask) where:
        # image: [1,Z,Y,X], mask: [Z,Y,X]
        image, mask = batch
        image = image.squeeze(0)  # [1,Z,Y,X]
        mask = mask.squeeze(0)    # [Z,Y,X]

        preds, logits_cpu = predict_volume(model, image, device, batch_slices=batch_slices)

        # Dice
        fg_d = mean_foreground_dice(preds, mask, num_classes)
        pc = dice_per_class(preds, mask, num_classes)
        pros_d = prostate_superclass_dice(preds, mask, prostate_ids=(4,5))
        prostate_dice_list.append(pros_d)

        mean_fg_dices.append(fg_d)
        for c in range(num_classes):
            per_class_dice_accum[c].append(pc[c])

        # Hierarchical costs (optional)
        if D is not None:
            h_arg = mean_hier_cost_argmax(preds, mask, D)
            # expected cost needs logits shaped [N,C,H,W] and targets [N,H,W]
            # Here N=Z (slices)
            logits = logits_cpu  # [Z,C,Y,X] on CPU
            targets = mask       # [Z,Y,X]
            h_exp = expected_hier_cost_from_logits(logits, targets, D)

            hcost_argmax_list.append(h_arg)
            hcost_expected_list.append(h_exp)

        per_case.append({
            "mean_fg_dice": fg_d,
            "per_class_dice": pc,
            "hcost_argmax": hcost_argmax_list[-1] if D is not None else None,
            "hcost_expected": hcost_expected_list[-1] if D is not None else None,
            "prostate_dice": pros_d,
        })

    # aggregate
    agg = {
        "mean_fg_dice": float(np.mean(mean_fg_dices)) if mean_fg_dices else 0.0,
        "per_class_dice_mean": [],
        "per_class_dice_std": [],
        "prostate_dice_mean": float(np.mean(prostate_dice_list)) if prostate_dice_list else None,
        "prostate_dice_std": float(np.std(prostate_dice_list)) if prostate_dice_list else None
    }

    for c in range(num_classes):
        vals = np.array(per_class_dice_accum[c], dtype=np.float64)
        # ignore nan (class absent)
        vals = vals[np.isfinite(vals)]
        agg["per_class_dice_mean"].append(float(np.mean(vals)) if vals.size else float("nan"))
        agg["per_class_dice_std"].append(float(np.std(vals)) if vals.size else float("nan"))

    if D is not None and hcost_argmax_list:
        agg["hcost_argmax_mean"] = float(np.mean(hcost_argmax_list))
        agg["hcost_argmax_std"] = float(np.std(hcost_argmax_list))
        agg["hcost_expected_mean"] = float(np.mean(hcost_expected_list))
        agg["hcost_expected_std"] = float(np.std(hcost_expected_list))
    else:
        agg["hcost_argmax_mean"] = None
        agg["hcost_expected_mean"] = None

    return agg, per_case


def main():
    ap = argparse.ArgumentParser(description="Test flat vs hierarchical models on full NIfTI volumes.")
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Path to split dir containing images/ and masks/ (e.g. data/preprocessed_data/test)")
    ap.add_argument("--flat_ckpt", type=str, required=True)
    ap.add_argument("--hier_ckpt", type=str, required=True)
    ap.add_argument("--num_classes", type=int, default=9)
    ap.add_argument("--batch_slices", type=int, default=8)
    ap.add_argument("--use_hierarchical_metrics", action="store_true",
                    help="If set, compute hierarchical costs using distance matrix D.")
    ap.add_argument("--out_json", type=str, default="test_metrics.json")
    ap.add_argument("--out_csv", type=str, default="test_metrics.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = MalePelvicDataset(args.data_dir)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    D = None
    if args.use_hierarchical_metrics:
        D = build_distance_matrix(args.num_classes, device=torch.device("cpu"))  # keep on CPU for indexing

    # Load models
    flat_model = load_checkpoint(UNet(in_channels=1, num_classes=args.num_classes), args.flat_ckpt, device)
    hier_model = load_checkpoint(UNet(in_channels=1, num_classes=args.num_classes), args.hier_ckpt, device)

    # Evaluate
    flat_agg, _ = evaluate_model_on_test(flat_model, test_loader, device, args.num_classes, D=D, batch_slices=args.batch_slices)
    hier_agg, _ = evaluate_model_on_test(hier_model, test_loader, device, args.num_classes, D=D, batch_slices=args.batch_slices)

    results = {
        "flat": flat_agg,
        "hier": hier_agg,
        "num_cases": len(test_ds),
    }

    # Save JSON
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Saved JSON: {out_json}")

    # Save CSV (summary rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mean_fg_dice", "prostate_dice", "hcost_argmax_mean", "hcost_expected_mean"])
        writer.writerow(["flat", flat_agg["mean_fg_dice"], flat_agg["prostate_dice_mean"],
                 flat_agg["hcost_argmax_mean"], flat_agg["hcost_expected_mean"]])
        writer.writerow(["hier", flat_agg["mean_fg_dice"], flat_agg["prostate_dice_mean"],
                 flat_agg["hcost_argmax_mean"], flat_agg["hcost_expected_mean"]])
    print(f"Saved CSV:  {out_csv}")

    # Print quick summary
    print("\n=== TEST SUMMARY ===")
    print(f"Flat mean foreground Dice: {flat_agg['mean_fg_dice']:.4f}")
    print(f"Hier mean foreground Dice: {hier_agg['mean_fg_dice']:.4f}")
    if args.use_hierarchical_metrics:
        print(f"Flat expected h-cost:      {flat_agg['hcost_expected_mean']:.4f}")
        print(f"Hier expected h-cost:      {hier_agg['hcost_expected_mean']:.4f}")
        print(f"Flat prostate Dice:        {flat_agg['prostate_dice_mean']:.4f}")
        print(f"Hier prostate Dice:        {hier_agg['prostate_dice_mean']:.4f}")

if __name__ == "__main__":
    main()
