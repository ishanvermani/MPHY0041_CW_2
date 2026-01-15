import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path):
	with open(path, "r") as f:
		return json.load(f)


def extract_h_conf(data, model: str | None = None, split: str | None = None, h_key: str = "h_conf"):
	"""
	Return (H, title_suffix).

	Handles three cases:
	1) dict with top-level h_conf
	2) dict with flat/hier entries each containing h_conf
	3) legacy list of epoch dicts with split-specific key (train_h_conf/val_h_conf)
	"""

	# case 1: direct h_conf
	if isinstance(data, dict) and h_key in data:
		return data[h_key], data.get("epoch", "?")

	# case 2: test metrics with flat/hier blocks
	if isinstance(data, dict) and model and model in data and isinstance(data[model], dict):
		entry = data[model]
		if h_key in entry:
			return entry[h_key], entry.get("epoch", model)

	# case 3: legacy training metrics list
	if isinstance(data, list):
		if not split:
			print("split must be provided when metrics is a list of epochs")
		key = f"{split}_h_conf"
		for m in reversed(data):
			if key in m:
				return m[key], m.get("epoch", "?")

	print("Could not find h_conf in provided metrics")


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
	parser = argparse.ArgumentParser()
	parser.add_argument("--metrics", type=Path, required=True)
	parser.add_argument("--model", choices=["flat", "hier"], default="hier")
	parser.add_argument("--split", choices=["train", "val"], default=None,
				help="Only needed for legacy training metrics list; ignored for test-style files")
	parser.add_argument("--h_key", type=str, default="h_conf")
	parser.add_argument("--out", type=Path, default=Path("h_conf.png"))
	parser.add_argument("--class_names", type=str, nargs="*", default=None)
	args = parser.parse_args()

	if args.metrics.suffix.lower() == ".npy":
		H = np.load(args.metrics)
		title_suffix = args.model
	else:
		metrics = load_metrics(args.metrics)
		H, title_suffix = extract_h_conf(metrics, model=args.model, split=args.split, h_key=args.h_key)
		H = np.array(H, dtype=float)

	plot_heatmap(H, args.out, title=f"H ({title_suffix})", class_names=args.class_names)
	print(f"Saved heatmap to {args.out}")

if __name__ == "__main__":
	main()
