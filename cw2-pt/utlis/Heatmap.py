"""Plot hierarchical confusion matrix H as a heatmap.

Usage examples:
  python Heatmap.py --metrics metrics/metrics.json --split val --out h_conf.png
  python Heatmap.py --metrics metrics/metrics.json --split train --key val_super_dice

By default it picks the entry with the highest `val_super_dice` (fallback to `val_dice`),
pulls the chosen split's H (train_h_conf or val_h_conf), and plots it.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path):
	with open(path, "r") as f:
		return json.load(f)


def pick_best(metrics, key_candidates):
	for key in key_candidates:
		if all(key in m for m in metrics):
			return max(metrics, key=lambda m: m.get(key, float("-inf")))
	return metrics[-1]  # fallback: last epoch


def plot_heatmap(H: np.ndarray, out: Path, title: str = "Hierarchical Confusion", class_names=None):
	plt.figure(figsize=(6, 5))
	im = plt.imshow(H, cmap="magma")
	plt.colorbar(im, fraction=0.046, pad=0.04, label="weighted count")
	if class_names is None:
		class_names = [str(i) for i in range(H.shape[0])]
	plt.xticks(range(H.shape[1]), class_names, rotation=45, ha="right")
	plt.yticks(range(H.shape[0]), class_names)
	plt.xlabel("predicted")
	plt.ylabel("true")
	plt.title(title)
	plt.tight_layout()
	out.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(out, dpi=200)
	plt.close()


def main():
	parser = argparse.ArgumentParser(description="Plot hierarchical confusion H as heatmap")
	parser.add_argument("--metrics", type=Path, required=True, help="Path to metrics.json")
	parser.add_argument("--split", choices=["train", "val"], default="val", help="Which split's H to plot")
	parser.add_argument("--out", type=Path, default=Path("h_conf.png"), help="Output image path")
	parser.add_argument("--key", type=str, nargs="*", default=["val_super_dice", "val_dice"],
					help="Keys to select best epoch (in priority order)")
	args = parser.parse_args()

	metrics = load_metrics(args.metrics)
	best = pick_best(metrics, args.key)

	H_key = f"{args.split}_h_conf"
	H = best.get(H_key)
	if H is None:
		raise ValueError(f"No {H_key} found in selected metrics entry")

	H_arr = np.array(H, dtype=float)
	plot_heatmap(H_arr, args.out, title=f"{args.split} H (epoch {best.get('epoch', '?')})")
	print(f"Saved heatmap to {args.out}")


if __name__ == "__main__":
	main()
