#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N hierarchy_unet_50ep
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

$PY train.py \
  --data_dir data/preprocessed_data \
  --use_hierarchical_loss \
  --epochs 5 \
  --batch_size 1 \
  --lr 1e-3 \
  --num_classes 9 \
  --samples_per_volume 8 \
  --foreground_prob 0.8 \
  --num_workers 4 \
  --metrics_path metrics/
