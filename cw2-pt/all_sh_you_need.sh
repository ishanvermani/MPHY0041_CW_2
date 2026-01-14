#!/bin/bash

#############################################
# Inference / evaluation shortcuts
#############################################

# Show best model dice, assuming best_model.pt exists
python inference.py best-score --ckpt model_weights/best_model.pt

# Evaluate models (requires images/ and masks/ under data_dir)
python inference.py eval --data_dir split_data/test \
		--flat_ckpt model_weights/best_model_flat.pt \
		--hier_ckpt model_weights/best_model_hier.pt \
		--num_classes 9 --use_hierarchical_metrics \
		--out_json test_metrics.json --out_csv test_metrics.csv

# Plot Dice and Loss Curves
python inference.py plot --metrics_path data/preprocessed_data/metrics.csv --out plot.png

# Run predictions and overlays
python inference.py display --image_nii path/to/img.nii.gz --mask_nii path/to/mask.nii.gz \
		--flat_ckpt data/preprocessed_data/best_flat.pt --hier_ckpt data/preprocessed_data/best_hier.pt \
		--num_classes 9 --slices mid --out_dir prediction_masks


#############################################
# Train flat U-Net on GPU (SGE job script)
#############################################
# Submit this section with qsub if using Myriad; logs go to logs/flat_unet_with_super_dice.*
# Adjust PY to your environment path.

#$ -S /bin/bash
#$ -cwd
#$ -N flat_unet_with_super_dice
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#$ -l h_rt=06:00:00
#$ -l mem=16G
#$ -pe smp 4
#$ -l gpu=1
#$ -M rmapcag@ucl.ac.uk
#$ -m abe

set -euo pipefail

mkdir -p logs

PY=/myriadfs/home/rmapcag/y/envs/cw2-pt/bin/python

$PY train.py \
	--data_dir data/preprocessed_data \
	--epochs 50 \
	--batch_size 1 \
	--lr 1e-3 \
	--num_classes 9 \
	--samples_per_volume 8 \
	--foreground_prob 0.8 \
	--num_workers 4 \
	--metrics_path metrics/


#############################################
# Train hierarchical U-Net on GPU (SGE job script)
#############################################
# Submit with qsub; logs go to logs/hierarchy_unet_50ep.*; adjust PY path as needed.

#$ -S /bin/bash
#$ -cwd
#$ -N hierarchy_unet_50ep
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#$ -l h_rt=08:00:00
#$ -l mem=16G
#$ -pe smp 4
#$ -l gpu=1
#$ -M rmapcag@ucl.ac.uk
#$ -m abe

set -euo pipefail

mkdir -p logs

PY=/myriadfs/home/rmapcag/y/envs/cw2-pt/bin/python

$PY train.py \
	--data_dir data/preprocessed_data \
	--use_hierarchical_loss \
	--epochs 50 \
	--batch_size 1 \
	--lr 1e-3 \
	--num_classes 9 \
	--samples_per_volume 8 \
	--foreground_prob 0.8 \
	--num_workers 4 \
	--metrics_path metrics/
