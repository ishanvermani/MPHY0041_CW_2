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
