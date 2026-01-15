import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MalePelvicDataset, PatchDataset
from src.model import UNet
from src.losses import ce_loss, hierarchical_ce_loss, build_distance_matrix
from utils.benchmark import (
	per_class_dice,
	hd95_per_class,
	hierarchy_confusion_fast,
	prostate_superclass_metrics,
)

def make_dataloaders(data_root: Path, batch_size: int, num_workers: int,
					 patch_size=(16, 192, 192), samples_per_volume: int = 8,
					 foreground_prob: float = 0.8):
	train_vols = MalePelvicDataset(str(data_root / "train"))
	val_vols = MalePelvicDataset(str(data_root / "val"))

	train_ds = PatchDataset(train_vols, patch_size=patch_size,
							foreground_prob=foreground_prob,
							samples_per_volume=samples_per_volume)
	val_ds = PatchDataset(val_vols, patch_size=patch_size,
						  foreground_prob=foreground_prob,
						  samples_per_volume=max(1, samples_per_volume // 2))

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
							  num_workers=num_workers, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
							num_workers=num_workers, pin_memory=True)
	return train_loader, val_loader


def reshape_to_2d(images, masks):
	"""
	Convert (B, C=1, Z, H, W) and (B, Z, H, W) into 2D slices for a 2D U-Net:
	returns (B*Z, C, H, W), (B*Z, H, W).
	"""
	b, c, z, h, w = images.shape
	images = images.permute(0, 2, 1, 3, 4).reshape(b * z, c, h, w)
	masks = masks.reshape(b * z, h, w)
	return images, masks

def expected_hierarchical_cost(logits: torch.Tensor, targets: torch.Tensor, D: torch.Tensor) -> float:
	""" Computes the mean expected hierarchical misclassification cost using softmax probs and a class-distance matrix D. """
	probs = torch.softmax(logits, dim=1)

	# Select the cost row corresponding to the true class
	D_row = D[targets]

	# Expected cost per pixel
	e_cost = (probs.permute(0,2,3,1) * D_row).sum(dim=-1)

	return e_cost.mean().item()


def dice_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6) -> float:
	""" Compute mean Dice across classes present in targets. """
	dice_sum, count = 0.0, 0
	for c in range(1, num_classes): # Ignore background when calculating dice score 
		pred_c = (preds == c)
		tgt_c = (targets == c)
		denom = pred_c.sum() + tgt_c.sum()
		if denom == 0:
			continue
		inter = (pred_c & tgt_c).sum()
		dice = (2.0 * inter.float() + eps) / (denom.float() + eps)
		dice_sum += dice
		count += 1
	if count == 0:
		return 0.0
	return (dice_sum / count).item()


def train_one_epoch(model, loader, optimizer, device, loss_fn, num_classes, D=None, alpha=None):
	""" Runs one training epoch and returns average loss, mean Dice, and superclass Dice. """
	model.train()
	total_loss, total_px, total_dice, dice_count = 0.0, 0, 0.0, 0
	super_dice_sum, super_dice_count = 0.0, 0

	for images, masks in loader:
		images, masks = images.to(device), masks.to(device)
		images, masks = reshape_to_2d(images, masks)

		optimizer.zero_grad()
		logits = model(images)
		loss = loss_fn(logits, masks, alpha)
		loss.backward()
		optimizer.step()

		total_loss += loss.item() * masks.numel()
		preds = logits.argmax(dim=1)
		total_px += masks.numel()

		# Dice on predicted labels
		dice = dice_score(preds.detach(), masks, num_classes)
		total_dice += dice
		dice_count += 1

		# Prostate superclass dice
		super_dice = prostate_superclass_metrics(logits.detach(), masks, num_classes)
		super_dice_sum += super_dice
		super_dice_count += 1

	avg_loss = total_loss / max(1, total_px)
	mean_dice = total_dice / max(1, dice_count)
	avg_super_dice = super_dice_sum / max(1, super_dice_count)
	return avg_loss, mean_dice, avg_super_dice

def evaluate(model, loader, device, loss_fn, num_classes, D=None, alpha=None):
	""" Evaluates the model and returns loss, Dice, hierarchical cost (optional), and superclass Dice. """
	model.eval()
	total_loss, total_px, total_dice, dice_count = 0.0, 0, 0.0, 0
	total_hcost, hcost_count = 0.0, 0 
	super_dice_sum, super_dice_count = 0.0, 0
	
	with torch.no_grad():
		for images, masks in loader:
			images, masks = images.to(device), masks.to(device)
			images, masks = reshape_to_2d(images, masks)
			logits = model(images)
			loss = loss_fn(logits, masks, alpha)

			total_loss += loss.item() * masks.numel()
			preds = logits.argmax(dim=1)
			total_px += masks.numel()

 			# Compute hierarchical expected cost if enabled
			if D is not None:
				h_cost = expected_hierarchical_cost(logits, masks, D)
				total_hcost += h_cost
				hcost_count += 1

			# Dice on predicted labels
			dice = dice_score(preds, masks, num_classes)
			total_dice += dice
			dice_count += 1

			# Prostate superclass dice
			super_dice = prostate_superclass_metrics(logits, masks, num_classes)
			super_dice_sum += super_dice
			super_dice_count += 1

	avg_loss = total_loss / max(1, total_px)
	mean_dice = total_dice / max(1, dice_count)

	avg_super_dice = super_dice_sum / max(1, super_dice_count)

	if D is not None:
		mean_hcost = total_hcost / max(1, hcost_count)
		return avg_loss, mean_dice, mean_hcost, avg_super_dice

	return avg_loss, mean_dice, None, avg_super_dice

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, required=True)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--training_epochs", type=int, default=1)
	parser.add_argument("--alpha", type=int, default=None)
	parser.add_argument("--batch_size", type=int, default=1)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--num_classes", type=int, default=9)
	parser.add_argument("--use_hierarchical_loss", action="store_true", help="Use anatomy-aware hierarchical cross-entropy")
	parser.add_argument("--samples_per_volume", type=int, default=8)
	parser.add_argument("--foreground_prob", type=float, default=0.8)
	parser.add_argument("--patch_z", type=int, default=16)
	parser.add_argument("--patch_y", type=int, default=192)
	parser.add_argument("--patch_x", type=int, default=192)
	parser.add_argument("--metrics_path", type=str, default=None)
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	data_root = Path(args.data_dir)

	patch_size = (args.patch_z, args.patch_y, args.patch_x)
	train_loader, val_loader = make_dataloaders(
		data_root,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		patch_size=patch_size,
		samples_per_volume=args.samples_per_volume,
		foreground_prob=args.foreground_prob,
	)

	model = UNet(in_channels=1, num_classes=args.num_classes).to(device)

	# Choose loss function
	if args.use_hierarchical_loss:
		D = build_distance_matrix(args.num_classes, device)

		def loss_fn(logits, targets, alpha):
			# Clamp invalid labels to valid class range
			if targets.max() >= args.num_classes:
				targets = targets.clamp_max(args.num_classes - 1)
			return hierarchical_ce_loss(logits, targets, D, alpha)
	else:
		def loss_fn(logits, targets, alpha):
			if targets.max() >= args.num_classes:
				targets = targets.clamp_max(args.num_classes - 1)
			return ce_loss(logits, targets)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	metrics_base = Path(args.metrics_path) if args.metrics_path else data_root / "metrics"
	metrics_json = metrics_base.with_suffix(".json")
	metrics_csv = metrics_base.with_suffix(".csv")
	metrics = []

	def save_metrics():
		metrics_json.parent.mkdir(parents=True, exist_ok=True)
		with open(metrics_json, "w") as f:
			json.dump(metrics, f, indent=2)
		with open(metrics_csv, "w", newline="") as f:
			writer = csv.DictWriter(f, fieldnames=[
				"epoch", "train_loss", "train_dice", "train_super_dice",
				"val_loss", "val_dice", "val_super_dice"
			], extrasaction="ignore")
			writer.writeheader()
			for row in metrics:
				writer.writerow(row)
		return metrics_json, metrics_csv
	
	training_val_dice = []

	# Alpha grid search if hierarchical loss is used 
	if args.use_hierarchical_loss and args.alpha is None:

		print(f"Starting hyperparam training for {args.training_epochs} epochs on {device}…")

		alpha_gridsearch = [1, 2, 3, 5, 10]
		initial_state = model.state_dict()

		for a in alpha_gridsearch:

			model.load_state_dict(initial_state)
			optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
			t0 = time.time()
			
			print(f"Training alpha {a} for {args.training_epochs} epochs")
			for epoch in range(args.training_epochs):
			
				train_loss, train_dice, train_super_dice = train_one_epoch(
					model, train_loader, optimizer, device, loss_fn, args.num_classes, D if args.use_hierarchical_loss else None, a if args.use_hierarchical_loss else None)

			# Evaluate alpha candidate
			val_loss, val_dice, val_hcost, val_super_dice = evaluate(
			 	model, val_loader, device, loss_fn, args.num_classes, D if args.use_hierarchical_loss else None, a if args.use_hierarchical_loss else None)
			dt = time.time() - t0

			print(f"Alpha {a:03d} | {dt:5.1f}s | "
				f"val loss {val_loss:.4f} dice {val_dice:.4f}" )
			
			training_val_dice.append(val_dice)

		# Select best alpha
		alpha = alpha_gridsearch[np.argmax(training_val_dice)]

		print(f"Selected alpha: {alpha}")
		
		model.load_state_dict(initial_state)

	else:
		alpha = args.alpha
		print(f"Using provided alpha: {alpha}")

	# Full training loop 
	print(f"Starting training for {args.epochs} epochs on {device}…")
	best_super = -float("inf")
	best_val_dice = -float("inf")
	try:
		for epoch in range(1, args.epochs + 1):
			t0 = time.time()

			train_loss, train_dice, train_super_dice = train_one_epoch(
				model, train_loader, optimizer, device, loss_fn, args.num_classes, D if args.use_hierarchical_loss else None, alpha if args.use_hierarchical_loss else None)

			val_loss, val_dice, val_hcost, val_super_dice = evaluate(
			 	model, val_loader, device, loss_fn, args.num_classes, D if args.use_hierarchical_loss else None, alpha if args.use_hierarchical_loss else None)

			dt = time.time() - t0

			print(f"Epoch {epoch:03d} | {dt:5.1f}s | "
				  f"train loss {train_loss:.4f} dice {train_dice:.4f} superDice {train_super_dice:.4f} | "
				  f"val loss {val_loss:.4f} dice {val_dice:.4f} superDice {val_super_dice:.4f}")

			metrics.append({
				"epoch": epoch,
				"train_loss": train_loss,
				"train_dice": train_dice,
				"train_super_dice": train_super_dice,
				"val_loss": val_loss,
				"val_dice": val_dice,
				"h_cost": val_hcost,
				"val_super_dice": val_super_dice
			})

			mj, mc = save_metrics()
			print(f"   Logged metrics to {mj} and {mc}")

			# Save best model by superclass dice
			if val_super_dice > best_super:
				best_super = val_super_dice
				ckpt_path = data_root / "best_super_dice_model.pt"
				torch.save({"model_state": model.state_dict(),
						"epoch": epoch,
						"val_super_dice": val_super_dice,
						"val_dice": val_dice,
						"val_loss": val_loss}, ckpt_path)
				print(f"	   Saved checkpoint (best superDice) to {ckpt_path}")

			# Save best model by overall dice
			if val_dice > best_val_dice:
				best_val_dice = val_dice
				dice_ckpt = data_root / "best_dice_model.pt"
				torch.save({"model_state": model.state_dict(),
						"epoch": epoch,
						"val_dice": val_dice,
						"val_super_dice": val_super_dice,
						"val_loss": val_loss}, dice_ckpt)
				print(f"	   Saved checkpoint (best overall Dice) to {dice_ckpt}")
	except KeyboardInterrupt:
		print("\nKeyboardInterrupt received. Exiting training loop.")

if __name__ == "__main__":
	main()
