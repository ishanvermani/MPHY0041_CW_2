# Show best model dice, assuming best_model.pt exists
python inference.py best-score --ckpt data/processed/best_model.pt

# Evaluate models (requires images/ and masks/ under data_dir)
python inference.py eval --data_dir data/preprocessed_data/test \
    --flat_ckpt data/preprocessed_data/best_flat.pt \
    --hier_ckpt data/preprocessed_data/best_hier.pt \
    --num_classes 9 --use_hierarchical_metrics \
    --out_json test_metrics.json --out_csv test_metrics.csv

# Plot Dice and Loss Curves
python inference.py plot --metrics_path data/preprocessed_data/metrics.csv --out plot.png

# Run predictions and overlays
python inference.py display --image_nii path/to/img.nii.gz --mask_nii path/to/mask.nii.gz \
    --flat_ckpt data/preprocessed_data/best_flat.pt --hier_ckpt data/preprocessed_data/best_hier.pt \
    --num_classes 9 --slices mid --out_dir prediction_masks
    