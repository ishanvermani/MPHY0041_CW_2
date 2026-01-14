"""Evaluation utilities: per-class Dice, per-class HD95, hierarchical confusion.

These functions are imported by train.py to keep training loop slim.
Requires scipy for HD95.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_per_class(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6):
	"""
	3D volume per-class Dice. preds/targets: [Z,Y,X]; returns list length num_classes (nan if class absent).
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
	"""Mean Dice across present foreground classes (1..C-1), ignore background."""
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
	"""Binary Dice; returns 1.0 if both empty."""
	inter = (pred_bin & tgt_bin).sum()
	denom = pred_bin.sum() + tgt_bin.sum()
	if denom == 0:
		return 1.0
	return ((2.0 * inter.float() + eps) / (denom.float() + eps)).item()


def prostate_superclass_dice(preds: torch.Tensor, targets: torch.Tensor, prostate_ids=(4, 5)) -> float:
	"""Merge specified prostate ids into one mask and compute binary Dice (prostate vs rest)."""
	pred_p = torch.zeros_like(preds, dtype=torch.bool)
	tgt_p = torch.zeros_like(targets, dtype=torch.bool)
	for pid in prostate_ids:
		pred_p |= (preds == pid)
		tgt_p |= (targets == pid)
	return binary_dice(pred_p, tgt_p)

@torch.no_grad()
def expected_hier_cost_from_logits(logits: torch.Tensor, targets: torch.Tensor, D: torch.Tensor) -> float:
	"""Soft/expected hierarchical cost using softmax probabilities."""
	probs = torch.softmax(logits, dim=1)              # [N,C,H,W]
	D_row = D[targets]                                # [N,H,W,C]
	exp_cost = (probs.permute(0, 2, 3, 1) * D_row).sum(dim=-1)  # [N,H,W]
	return exp_cost.mean().item()


def merge_prostate_superclass(preds: torch.Tensor, targets: torch.Tensor):
	"""Return merged copies where prostate sub-classes 1/2 (index 4, 5) are mapped to 4.

	Background stays 0; other classes remain unchanged. Shapes preserved.
	"""
	merged = preds.clone()
	merged_tgt = targets.clone()
	
	merged[merged == 5] = 4
	merged_tgt[merged_tgt == 5] = 4
	return merged, merged_tgt


def binary_dice(pred_mask: torch.Tensor, tgt_mask: torch.Tensor, eps: float = 1e-6) -> float:
	"""Dice for binary masks (bool/int)."""
	inter = (pred_mask & tgt_mask).sum().float()
	denom = pred_mask.sum().float() + tgt_mask.sum().float()
	if denom == 0:
		return float("nan")
	return ((2 * inter + eps) / (denom + eps)).item()


def per_class_dice(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6):
	"""Return list of Dice per class; classes absent in both pred/target get nan."""
	res = []
	for c in range(num_classes):
		pred_c = (preds == c)
		tgt_c = (targets == c)
		denom = pred_c.sum() + tgt_c.sum()
		if denom == 0:
			res.append(float("nan"))
			continue
		inter = (pred_c & tgt_c).sum()
		dice = (2.0 * inter.float() + eps) / (denom.float() + eps)
		res.append(dice.item())
	return res


def hd95_single(pred_mask: np.ndarray, tgt_mask: np.ndarray, spacing=(1.0, 1.0)):
	"""Symmetric 95th percentile Hausdorff distance for binary masks.

	Returns nan if both empty; inf if only one is empty.
	"""
	if pred_mask.sum() == 0 and tgt_mask.sum() == 0:
		return float("nan")
	if pred_mask.sum() == 0 or tgt_mask.sum() == 0:
		return float("inf")

	pred_surf = pred_mask ^ binary_erosion(pred_mask)
	tgt_surf = tgt_mask ^ binary_erosion(tgt_mask)

	dist_tgt = distance_transform_edt(~tgt_mask, sampling=spacing)
	dist_pred = distance_transform_edt(~pred_mask, sampling=spacing)

	d_a_to_b = dist_tgt[pred_surf]
	d_b_to_a = dist_pred[tgt_surf]
	if d_a_to_b.size == 0 or d_b_to_a.size == 0:
		return float("inf")
	return float(np.percentile(np.concatenate([d_a_to_b, d_b_to_a]), 95))


def hd95_per_class(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, spacing=(1.0, 1.0)):
	"""Compute per-class HD95 over stacked batch of 2D slices."""
	res = []
	p_np = preds.detach().cpu().numpy()
	t_np = targets.detach().cpu().numpy()
	for c in range(num_classes):
		vals = []
		for i in range(p_np.shape[0]):
			pred_c = (p_np[i] == c)
			tgt_c = (t_np[i] == c)
			v = hd95_single(pred_c, tgt_c, spacing=spacing)
			if np.isfinite(v):
				vals.append(v)
		
		if len(vals) == 0:
			res.append(float("nan"))
		else:
			res.append(float(np.median(vals)))
	return res

def hierarchy_confusion_fast(preds: torch.Tensor, targets: torch.Tensor, D: torch.Tensor):
	"""Weighted confusion accumulation using distance matrix D (rows=true, cols=pred)."""
	num_classes = D.shape[0]
	preds = preds.flatten()
	targets = targets.flatten()
	weights = D[targets, preds]
	H = torch.zeros((num_classes, num_classes), device=preds.device)
	H.index_put_((targets, preds), weights, accumulate=True)
	return H


def prostate_superclass_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int):
	"""Compute merged-prostate Dice and AUC (prostate vs rest) without altering base preds.

	- Dice uses argmax preds merged (4/5 -> 4) and class-1 Dice.
	- AUC uses softmax probs; positive when target in {4, 5}.
	"""
	with torch.no_grad():
		preds = logits.argmax(dim=1)
		merged_pred, merged_tgt = merge_prostate_superclass(preds, targets)
		prostate_dice = binary_dice(merged_pred == 4, merged_tgt == 4)		
	return prostate_dice
