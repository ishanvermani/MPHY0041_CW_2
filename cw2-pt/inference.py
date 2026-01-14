"""Unified inference/utility CLI.

Subcommands:
- best-score: print metadata from a saved checkpoint (epoch/val scores if present).
- eval: run full-volume evaluation (wraps utlis/score_train_val.py).
- plot: plot loss/dice curves from metrics CSV/JSON (wraps utlis/plot_loss_dice.py).
- display: run prediction/overlay export (wraps utlis/display_nii.py).
"""

import argparse
import json
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import MalePelvicDataset
from src.model import UNet
from src.losses import build_distance_matrix
from utils import test as stv
from utils import plot_loss_dice as pld
from utils import display_nii as dspy


# ---------------------------
# best-score
# ---------------------------

def cmd_best_score(args: argparse.Namespace):
	ckpt_path = Path(args.ckpt)
	try:
		checkpoint = torch.load(ckpt_path, map_location="cpu")
		if isinstance(checkpoint, dict) and "epoch" in checkpoint:
			print("---------------------------------------")
			print(f"Model saved at Epoch:  {checkpoint['epoch']}")
			print(f"Best Validation Loss:  {checkpoint['val_loss']:.4f}")
			print(f"Best Validation Dice:  {checkpoint['val_dice']:.4f}")
			print("---------------------------------------")
		else:
			print("This file contains only the model weights, no score data.")
	except FileNotFoundError:
		print(f"Could not find checkpoint at {ckpt_path}")
	except Exception as e:
		print(f"Error reading file: {e}")


# ---------------------------------------------
# Evaluate the model on train & validation sets
# ---------------------------------------------

def cmd_eval(args: argparse.Namespace):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	test_ds = MalePelvicDataset(args.data_dir)
	test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

	D = None
	if args.use_hierarchical_metrics:
		D = build_distance_matrix(args.num_classes, device=torch.device("cpu"))

	results = {}

	if args.flat_ckpt:
		flat_model = stv.load_checkpoint(UNet(in_channels=1, num_classes=args.num_classes), args.flat_ckpt, device)
		flat_agg, flat_cases = stv.evaluate_model_on_test(flat_model, test_loader, device, args.num_classes, D=D, batch_slices=args.batch_slices)
		results["flat"] = {"agg": flat_agg, "cases": flat_cases}

	if args.hier_ckpt:
		hier_model = stv.load_checkpoint(UNet(in_channels=1, num_classes=args.num_classes), args.hier_ckpt, device)
		hier_agg, hier_cases = stv.evaluate_model_on_test(hier_model, test_loader, device, args.num_classes, D=D, batch_slices=args.batch_slices)
		results["hier"] = {"agg": hier_agg, "cases": hier_cases}

	if not results:
		raise RuntimeError("No checkpoints provided. Pass --flat_ckpt and/or --hier_ckpt.")

	# Save JSON summary
	out_json = Path(args.out_json)
	out_json.parent.mkdir(parents=True, exist_ok=True)
	to_dump = {k: v["agg"] for k, v in results.items()}
	to_dump["num_cases"] = len(test_ds)
	out_json.write_text(json.dumps(to_dump, indent=2))
	print(f"Saved JSON: {out_json}")

	# Save CSV summary (one row per model)
	out_csv = Path(args.out_csv)
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	headers = ["model", "mean_fg_dice", "prostate_dice", "hcost_expected_mean"]
	rows = []
	for name, obj in results.items():
		agg = obj["agg"]
		rows.append([name, agg.get("mean_fg_dice"), agg.get("prostate_dice_mean"), agg.get("hcost_expected_mean")])
	with open(out_csv, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(headers)
		writer.writerows(rows)
	print(f"Saved CSV:  {out_csv}")

	# Console summary
	print("\n=== Evaluation SUMMARY ===")
	for name, obj in results.items():
		agg = obj["agg"]
		print(f"{name.capitalize()} mean foreground Dice: {agg.get('mean_fg_dice', float('nan')):.4f}")
		if args.use_hierarchical_metrics:
			print(f"{name.capitalize()} expected h-cost:      {agg.get('hcost_expected_mean', float('nan')):.4f}")
		print(f"{name.capitalize()} mean prostate Dice:        {agg.get('prostate_dice_mean', float('nan')):.4f}")


# ---------------------------
# Plot loss/dice curves
# ---------------------------

def cmd_plot(args: argparse.Namespace):
	metrics_path = Path(args.metrics_path)
	metrics = pld.load_metrics(metrics_path)
	if len(metrics) == 0:
		raise RuntimeError("Metrics file is empty—run training first.")
	out_path = Path(args.out) if args.out else None
	pld.plot_dice(metrics, out_path)


# ---------------------------------
# Display predictions on nii images
# ---------------------------------

def cmd_display(args: argparse.Namespace):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	img_nib, img_zyx = dspy.load_img_as_zyx(args.image_nii)
	gt_zyx = None
	if args.mask_nii is not None:
		_, gt_zyx = dspy.load_mask_as_zyx(args.mask_nii)

	flat_model = dspy.load_checkpoint(UNet(in_channels=1, num_classes=args.num_classes), args.flat_ckpt, device)
	hier_model = dspy.load_checkpoint(UNet(in_channels=1, num_classes=args.num_classes), args.hier_ckpt, device)

	pred_flat_zyx = dspy.predict_volume_zyx(flat_model, img_zyx, device, batch_slices=args.batch_slices)
	pred_hier_zyx = dspy.predict_volume_zyx(hier_model, img_zyx, device, batch_slices=args.batch_slices)

	dspy.save_pred_like_image(img_nib, pred_flat_zyx, str(out_dir / "pred_flat.nii.gz"))
	dspy.save_pred_like_image(img_nib, pred_hier_zyx, str(out_dir / "pred_hier.nii.gz"))

	slices = dspy.parse_slices(args.slices, Z=img_zyx.shape[0])
	dspy.overlay_pngs(img_zyx, pred_flat_zyx, pred_hier_zyx, gt_zyx, out_dir / "png_overlays", slices, alpha=args.alpha)

	print("Saved outputs:")
	print(f"  {out_dir / 'pred_flat.nii.gz'}")
	print(f"  {out_dir / 'pred_hier.nii.gz'}")
	print(f"  {out_dir / 'png_overlays'}")


# ---------------------------
# parser
# ---------------------------

def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Unified inference/utility entrypoint")
	subs = parser.add_subparsers(dest="command", required=True)

	# best-score
	p_best = subs.add_parser("best-score", help="Print checkpoint epoch/loss/dice if stored")
	p_best.add_argument("--ckpt", default="data/processed/best_model.pt", help="Path to checkpoint")
	p_best.set_defaults(func=cmd_best_score)

	# eval
	p_eval = subs.add_parser("eval", help="Evaluate checkpoints on a dataset (wraps score_train_val)")
	p_eval.add_argument("--data_dir", required=True, help="Dir containing images/ and masks/")
	p_eval.add_argument("--flat_ckpt", default=None, help="Path to flat model checkpoint")
	p_eval.add_argument("--hier_ckpt", default=None, help="Path to hierarchical model checkpoint")
	p_eval.add_argument("--num_classes", type=int, default=9)
	p_eval.add_argument("--batch_slices", type=int, default=8)
	p_eval.add_argument("--use_hierarchical_metrics", action="store_true")
	p_eval.add_argument("--out_json", default="test_metrics.json")
	p_eval.add_argument("--out_csv", default="test_metrics.csv")
	p_eval.set_defaults(func=cmd_eval)

	# plot
	p_plot = subs.add_parser("plot", help="Plot loss/dice curves from metrics JSON/CSV")
	p_plot.add_argument("--metrics_path", required=True, help="Path to metrics.json or metrics.csv")
	p_plot.add_argument("--out", default=None, help="Optional output path for plot (png)")
	p_plot.set_defaults(func=cmd_plot)

	# display
	p_disp = subs.add_parser("display", help="Run prediction and overlay PNGs/NIfTI outputs")
	p_disp.add_argument("--image_nii", required=True)
	p_disp.add_argument("--flat_ckpt", required=True)
	p_disp.add_argument("--hier_ckpt", required=True)
	p_disp.add_argument("--num_classes", type=int, default=9)
	p_disp.add_argument("--mask_nii", default=None)
	p_disp.add_argument("--out_dir", default="prediction_masks")
	p_disp.add_argument("--batch_slices", type=int, default=8)
	p_disp.add_argument("--slices", default="mid")
	p_disp.add_argument("--alpha", type=float, default=0.45)
	p_disp.set_defaults(func=cmd_display)

	return parser


def main():
	parser = build_parser()
	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()