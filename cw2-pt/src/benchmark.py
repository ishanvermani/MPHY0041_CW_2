"""Evaluation utilities: per-class Dice, per-class HD95, hierarchical confusion.

These functions are imported by train.py to keep training loop slim.
Requires scipy for HD95.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


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


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
	"""Compute ROC AUC for binary labels using torch ops. labels in {0,1}."""
	labels = labels.float()
	pos = labels.sum()
	neg = labels.numel() - pos
	if pos == 0 or neg == 0:
		return float("nan")
	# sort by score descending
	scores, idx = torch.sort(scores, descending=True)
	sorted_labels = labels[idx]
	tps = torch.cumsum(sorted_labels, dim=0)
	fps = torch.cumsum(1 - sorted_labels, dim=0)
	# prepend (0,0)
	tps = torch.cat([torch.zeros(1, device=tps.device), tps])
	fps = torch.cat([torch.zeros(1, device=fps.device), fps])
	# TPR/FPR
	tpr = tps / pos
	fpr = fps / neg
	# trapezoidal rule
	dfpr = fpr[1:] - fpr[:-1]
	auc = torch.sum(dfpr * (tpr[1:] + tpr[:-1]) * 0.5)
	return auc.item()


def prostate_superclass_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int):
	"""Compute merged-prostate Dice and AUC (prostate vs rest) without altering base preds.

	- Dice uses argmax preds merged (4/5 -> 4) and class-1 Dice.
	- AUC uses softmax probs; positive when target in {4, 5}.
	"""
	with torch.no_grad():
		preds = logits.argmax(dim=1)
		merged_pred, merged_tgt = merge_prostate_superclass(preds, targets)
		prostate_dice = binary_dice(merged_pred == 4, merged_tgt == 4)

		probs = torch.softmax(logits, dim=1)
		prostate_score = probs[:, [4, 5]].sum(dim=1)
		labels = (targets == 4) | (targets == 5)
		prostate_auc = binary_auc(prostate_score.flatten(), labels.flatten())
	return prostate_dice, prostate_auc

