#!/bin/bash

#############################################
# Inference for Training Evaluation
#############################################

# Show the best model's dice on different metrics
python inference.py best-score --ckpt metrics/metrics_flat_super_dice.csv



# Plot Dice and Loss Curves for training process
python inference.py plot --metrics_path data/preprocessed_data/metrics.csv --out plot.png

# Run predictions and overlays
python inference.py display --image_nii path/to/img.nii.gz --mask_nii path/to/mask.nii.gz \
		--flat_ckpt data/preprocessed_data/best_flat.pt --hier_ckpt data/preprocessed_data/best_hier.pt \
		--num_classes 9 --slices mid --out_dir prediction_masks

# Heatmap
python inference.py heatmap --h_conf metrics/h_conf_hier.json --out h_conf_hier.png --title "Hier H Conf (test)"



#############################################
# Inference for Test command
#############################################

# Run test evaluation on test set and store metrics
python test.py \
	--data_dir split_data/test \
	--flat_ckpt model_weights/best_model_flat.pt \
	--hier_ckpt model_weights/best_model_hier.pt \
	--num_classes 9 \
	--use_hierarchical_metrics \
	--save_h_conf metrics/h_conf_{model}.json \
	--out_json metrics/test_metrics.json \
	--out_csv metrics/test_metrics.csv

# Show NII predictions and overlays
python inference.py display --image_nii path/to/img.nii.gz --mask_nii path/to/mask.nii.gz \
		--flat_ckpt data/preprocessed_data/best_flat.pt --hier_ckpt data/preprocessed_data/best_hier.pt \
		--num_classes 9 --slices mid --out_dir prediction_masks

# Heatmap Visualization
python inference.py heatmap --h_conf metrics/h_conf_hier.json --out h_conf_hier.png --title "Hier H Conf (test)"



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

#!/bin/bash
#############################################
# Test/evaluate flat vs hierarchical U-Net on GPU (SGE job script)
#############################################
#$ -S /bin/bash
#$ -cwd
#$ -N test
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#$ -l h_rt=02:00:00
#$ -l mem=16G
#$ -pe smp 4
#$ -l gpu=1
#$ -M rmapcag@ucl.ac.uk
#$ -m abe

set -euo pipefail
mkdir -p logs

PY=/myriadfs/home/rmapcag/y/envs/cw2-pt/bin/python

$PY test.py \
  --data_dir data/preprocessed_data/test \
  --flat_ckpt model_weights/best_model_flat.pt \
  --hier_ckpt model_weights/best_model_heir.pt \
  --num_classes 9 \
  --use_hierarchical_metrics \
  --save_h_conf metrics/h_conf_{model}.npy
  --out_csv metrics/test_metrics.csv \
  --out_json metrics/test_metrics.json