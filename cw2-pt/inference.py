import argparse
import json
import csv
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import MalePelvicDataset
from src.model import UNet
from src.losses import build_distance_matrix
import test as stv
from utils import plot_loss_dice as pld
from utils import display_nii as dspy
from utils import Heatmap as hmap



# best-score
def cmd_best_score(args: argparse.Namespace):
	ckpt_path = Path(args.ckpt)

	# If user passes a csv, summarise best val_dice / val_super_dice
	if ckpt_path.suffix == ".csv":
		if not ckpt_path.exists():
			print(f"Could not find metrics CSV at {ckpt_path}")
			return
		try:
			with open(ckpt_path) as f:
				reader = csv.DictReader(f)
				rows = list(reader)
			if not rows:
				print(f"Metrics CSV {ckpt_path} is empty.")
				return
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
			print(f"Error reading CSV {ckpt_path}: {e}")
		return

	# Otherwise treat as checkpoint
	try:
		checkpoint = torch.load(ckpt_path, map_location="cpu")
		if isinstance(checkpoint, dict) and "epoch" in checkpoint:
			print("---------------------------------------")
			print(f"Model saved at Epoch:  {checkpoint['epoch']}")
			print(f"Best Validation Loss:  {checkpoint['val_loss']:.4f}")
			print(f"Best Validation Dice:  {checkpoint['val_dice']:.4f}")
			if "val_super_dice" in checkpoint:
				print(f"Best Validation Superclass Dice: {checkpoint['val_super_dice']:.4f}")
			print("---------------------------------------")
		else:
			print("This file contains only the model weights, no score data.")
	except FileNotFoundError:
		print(f"Could not find checkpoint at {ckpt_path}")
	except Exception as e:
		print(f"Error reading file: {e}")

# Evaluate the model on train & validation sets
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

	# Save json summary
	out_json = Path(args.out_json)
	out_json.parent.mkdir(parents=True, exist_ok=True)
	to_dump = {k: v["agg"] for k, v in results.items()}
	to_dump["num_cases"] = len(test_ds)
	out_json.write_text(json.dumps(to_dump, indent=2))
	print(f"Saved JSON: {out_json}")

	# Save csv summary (one row per model)
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

	print("\n=== Evaluation SUMMARY ===")
	for name, obj in results.items():
		agg = obj["agg"]
		print(f"{name.capitalize()} mean foreground Dice: {agg.get('mean_fg_dice', float('nan')):.4f}")
		if args.use_hierarchical_metrics:
			print(f"{name.capitalize()} expected h-cost:      {agg.get('hcost_expected_mean', float('nan')):.4f}")
		print(f"{name.capitalize()} mean prostate Dice:        {agg.get('prostate_dice_mean', float('nan')):.4f}")

# Plot loss/dice curves
def cmd_plot(args: argparse.Namespace):
	metrics_path = Path(args.metrics_path)
	metrics = pld.load_metrics(metrics_path)
	out_path = Path(args.out) if args.out else None
	pld.plot_dice(metrics, out_path)

# Display predictions on nii images
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

# Heatmap from saved H matrix
def load_h_conf_any(path: Path, model: str | None = None, split: str | None = None, h_key: str = "h_conf"):
	"""
	Load H from json or npy. Supports:
	- npy: direct matrix
	- json: direct {"h_conf": ...} or test metrics with flat/hier blocks, or list of epochs with train/val_h_conf
	"""
	if not path.exists():
		print(f"H confusion file not found: {path}")
	if path.suffix.lower() == ".npy":
		return np.load(path), model or path.stem
	if path.suffix.lower() == ".json":
		data = json.loads(path.read_text())
		H, title_suffix = hmap.extract_h_conf(data, model=model, split=split, h_key=h_key)
		return np.array(H, dtype=float), title_suffix

def cmd_heatmap(args: argparse.Namespace):
	H, title_suffix = load_h_conf_any(Path(args.metrics), model=args.model, split=args.split, h_key=args.h_key)
	hmap.plot_heatmap(H, Path(args.out), title=args.title or f"H ({title_suffix})", class_names=args.class_names)
	print(f"Saved heatmap to {args.out}")

def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser()
	subs = parser.add_subparsers(dest="command", required=True)

	# best-score
	p_best = subs.add_parser("best-score")
	p_best.add_argument("--ckpt", default="data/processed/best_model.pt")
	p_best.set_defaults(func=cmd_best_score)

	# eval
	p_eval = subs.add_parser("eval")
	p_eval.add_argument("--data_dir", required=True,)
	p_eval.add_argument("--flat_ckpt", default=None)
	p_eval.add_argument("--hier_ckpt", default=None)
	p_eval.add_argument("--num_classes", type=int, default=9)
	p_eval.add_argument("--batch_slices", type=int, default=8)
	p_eval.add_argument("--use_hierarchical_metrics", action="store_true")
	p_eval.add_argument("--out_json", default="test_metrics.json")
	p_eval.add_argument("--out_csv", default="test_metrics.csv")
	p_eval.set_defaults(func=cmd_eval)

	# plot
	p_plot = subs.add_parser("plot")
	p_plot.add_argument("--metrics_path", required=True)
	p_plot.add_argument("--out", default=None)
	p_plot.set_defaults(func=cmd_plot)

	# display
	p_disp = subs.add_parser("display")
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

	# heatmap
	p_hmap = subs.add_parser("heatmap")
	p_hmap.add_argument("--metrics", required=True)
	p_hmap.add_argument("--model", choices=["flat", "hier"], default="hier", help="Model block to use when metrics has flat/hier entries")
	p_hmap.add_argument("--split", choices=["train", "val"], default=None, help="Only for legacy epoch-list metrics (train/val_h_conf)")
	p_hmap.add_argument("--h_key", default="h_conf")
	p_hmap.add_argument("--out", default="h_conf.png")
	p_hmap.add_argument("--title", default=None)
	p_hmap.add_argument("--class_names", nargs="*", default=None)
	p_hmap.set_defaults(func=cmd_heatmap)

	return parser

def main():
	parser = build_parser()
	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()