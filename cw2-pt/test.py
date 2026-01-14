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
from utils.benchmark import (
    mean_foreground_dice,
    dice_per_class,
    prostate_superclass_dice,
    hd95_per_class,
    expected_hier_cost_from_logits,
    hierarchy_confusion_fast,
)


# ---------- Inference (slice-wise over Z) ----------

def _pad_to_multiple(x: torch.Tensor, multiple: int = 16):
    """Pad (N,1,H,W) to multiples of `multiple` using replicate padding; returns padded tensor and original (H,W)."""
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x_pad, (h, w)


@torch.no_grad()
def predict_volume(model: torch.nn.Module, image: torch.Tensor, device: torch.device, batch_slices: int = 8, multiple: int = 16):
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
        x_pad, (orig_h, orig_w) = _pad_to_multiple(x, multiple=multiple)
        logits = model(x_pad)  # [N,C,Hpad,Wpad]
        logits = logits[:, :, :orig_h, :orig_w]  # unpad back to original size
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

def evaluate_model_on_test(model, test_loader, device, num_classes, D=None, batch_slices=8, compute_h: bool = False):
    """
    Returns aggregated dict + per-case list
    """
    per_case = []

    # accumulators
    mean_fg_dices = []
    per_class_dice_accum = [[] for _ in range(num_classes)]
    per_class_hd95_accum = [[] for _ in range(num_classes)]
    hcost_expected_list = []
    prostate_dice_list = []
    hd95_list = []

    H_accum = None
    if compute_h and D is not None:
        H_accum = torch.zeros((num_classes, num_classes), device=torch.device("cpu"))


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

        #=============================================================
        # HD95
        #=============================================================

        hd95_vals = hd95_per_class(preds,mask,num_classes)
        valid_hd95 = [v for v in hd95_vals if not np.isnan(v) and not np.isinf(v)]
        avg_hd95 = np.mean(valid_hd95) if valid_hd95 else float("nan")
        
        #=============================================================


        mean_fg_dices.append(fg_d)
        for c in range(num_classes):
            per_class_dice_accum[c].append(pc[c])
            per_class_hd95_accum[c].append(hd95_vals[c])

        # overall HD95 accumulator
        hd95_list.append(avg_hd95)

        # Hierarchical costs (optional)
        if D is not None:
            # expected cost needs logits shaped [N,C,H,W] and targets [N,H,W]
            # Here N=Z (slices)
            logits = logits_cpu  # [Z,C,Y,X] on CPU
            targets = mask       # [Z,Y,X]
            h_exp = expected_hier_cost_from_logits(logits, targets, D)

            hcost_expected_list.append(h_exp)

            if H_accum is not None:
                H_accum += hierarchy_confusion_fast(preds, mask, D)

        per_case.append({
            "mean_fg_dice": fg_d,
            "per_class_dice": pc,
            "hcost_expected": hcost_expected_list[-1] if D is not None else None,
            "prostate_dice": pros_d,
            "hd95_mean": avg_hd95,
            "per_class_hd95": hd95_vals
        })

    # aggregate
    agg = {
        "mean_fg_dice": float(np.mean(mean_fg_dices)) if mean_fg_dices else 0.0,
        "per_class_dice_mean": [],
        "per_class_dice_std": [],
        "per_class_hd95_mean": [],
        "per_class_hd95_std": [],
        "prostate_dice_mean": float(np.mean(prostate_dice_list)) if prostate_dice_list else None,
        "prostate_dice_std": float(np.std(prostate_dice_list)) if prostate_dice_list else None,
        "hd95_mean": float(np.nanmean(hd95_list)) if hd95_list else None,
        "hd95_std": float(np.nanstd(hd95_list)) if hd95_list else None
    }

    for c in range(num_classes):
        vals = np.array(per_class_dice_accum[c], dtype=np.float64)
        # ignore nan (class absent)
        vals = vals[np.isfinite(vals)]
        agg["per_class_dice_mean"].append(float(np.mean(vals)) if vals.size else float("nan"))
        agg["per_class_dice_std"].append(float(np.std(vals)) if vals.size else float("nan"))

    hd95_vals_c = np.array(per_class_hd95_accum[c], dtype=np.float64)
    hd95_vals_c = hd95_vals_c[np.isfinite(hd95_vals_c)]
    agg["per_class_hd95_mean"].append(float(np.mean(hd95_vals_c)) if hd95_vals_c.size else float("nan"))
    agg["per_class_hd95_std"].append(float(np.std(hd95_vals_c)) if hd95_vals_c.size else float("nan"))

    if D is not None and hcost_expected_list:
        agg["hcost_expected_mean"] = float(np.mean(hcost_expected_list))
        agg["hcost_expected_std"] = float(np.std(hcost_expected_list))
    else:
        agg["hcost_expected_mean"] = None

    if H_accum is not None:
        agg["h_conf"] = H_accum.cpu().numpy().tolist()

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
    ap.add_argument("--save_h_conf", type=str, default=None,
                help="Optional path to save hierarchical confusion matrix (JSON or NPY). Use {model} to differentiate flat/hier.")
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
    flat_agg, _ = evaluate_model_on_test(
        flat_model,
        test_loader,
        device,
        args.num_classes,
        D=D,
        batch_slices=args.batch_slices,
        compute_h=bool(args.save_h_conf),
    )
    hier_agg, _ = evaluate_model_on_test(
        hier_model,
        test_loader,
        device,
        args.num_classes,
        D=D,
        batch_slices=args.batch_slices,
        compute_h=bool(args.save_h_conf),
    )

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
        writer.writerow(["model", "mean_fg_dice", "prostate_dice", "hcost_expected_mean"])
        writer.writerow(["flat", flat_agg["mean_fg_dice"], flat_agg["prostate_dice_mean"],
             flat_agg["hcost_expected_mean"]])
        writer.writerow(["hier", hier_agg["mean_fg_dice"], hier_agg["prostate_dice_mean"],
             hier_agg["hcost_expected_mean"]])
    print(f"Saved CSV:  {out_csv}")

    # Save hierarchical confusion (if requested and available)
    if args.save_h_conf and D is not None:
        h_path = Path(args.save_h_conf)
        for name, agg in [("flat", flat_agg), ("hier", hier_agg)]:
            if "h_conf" not in agg:
                continue
            if "{model}" in args.save_h_conf:
                out_path = Path(args.save_h_conf.format(model=name))
            else:
                out_path = h_path.with_name(f"{h_path.stem}_{name}{h_path.suffix}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.suffix.lower() == ".npy":
                np.save(out_path, np.array(agg["h_conf"], dtype=float))
            else:
                out_path.write_text(json.dumps({"h_conf": agg["h_conf"]}, indent=2))
            print(f"Saved H confusion for {name} to {out_path}")

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