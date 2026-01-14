import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import argparse

TARGET_SPACING = (0.75, 0.75, 3.0)

def normalize(img):
    lo, hi = np.percentile(img, (0.5, 99.5))
    img = np.clip(img, lo, hi)
    return (img - img.mean()) / (img.std() + 1e-8)

def resample(vol_zyx, current_spacing, target_spacing, islabel):
    cx, cy, cz = current_spacing
    tx, ty, tz = target_spacing
    zoom_factors = (cz / tz, cy / ty, cx / tx)
    order = 0 if islabel else 1
    return zoom(vol_zyx, zoom=zoom_factors, order=order)

def make_affine_like(old_affine, target_spacing_xyz):
    """
    Keep axis directions from old_affine but set voxel sizes to target spacing.
    """
    aff = old_affine.copy()
    R = aff[:3, :3]
    # normalize columns to get directions, then scale by spacing
    dirs = R / (np.linalg.norm(R, axis=0, keepdims=True) + 1e-12)
    aff[:3, :3] = dirs * np.array(target_spacing_xyz)[None, :]
    return aff

def preprocess_case(img_path, mask_path, out_image_path, out_mask_path):
    img_nib = nib.load(img_path)
    mask_nib = nib.load(mask_path)
    img = img_nib.get_fdata().astype(np.float32)
    mask = mask_nib.get_fdata().astype(np.float32)

    img_zyx = np.transpose(img, (2, 1, 0))
    mask_zyx = np.transpose(mask, (2, 1, 0))

    current_spacing = img_nib.header.get_zooms()[:3]

    img_zyx = resample(img_zyx, current_spacing, TARGET_SPACING, False)
    mask_zyx = resample(mask_zyx, current_spacing, TARGET_SPACING, True)

    img_zyx = normalize(img_zyx)

    img_xyz = np.transpose(img_zyx, (2, 1, 0))
    mask_xyz = np.transpose(mask_zyx, (2, 1, 0)).astype(np.uint8)

    new_affine = make_affine_like(img_nib.affine, TARGET_SPACING)

    nib.save(nib.Nifti1Image(img_xyz, new_affine), out_image_path)
    nib.save(nib.Nifti1Image(mask_xyz, new_affine), out_mask_path)

def image_file_to_mask_file(image_file):
    return image_file.replace("_img.nii.gz", "_mask.nii.gz").replace("_img.nii", "_mask.nii")

def compress_to_gzip(file):
    if file.endswith(".nii.gz"):
        return file
    if file.endswith(".nii"):
        return file + ".gz"
    return file + ".nii.gz"

def preprocess_split(split_dir, output_dir):
    
    input_image_dir = os.path.join(split_dir, "images")
    input_mask_dir = os.path.join(split_dir, "masks")

    output_image_dir = os.path.join(output_dir, "images")
    output_mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    for file in sorted(os.listdir(input_image_dir)):
        if not (file.endswith(".nii") or file.endswith(".nii.gz")):
            continue

        image_path = os.path.join(input_image_dir, file)
        mask_file = image_file_to_mask_file(file)
        mask_path = os.path.join(input_mask_dir, mask_file)
        if not os.path.exists(mask_path):
            print(f"Mask not found for {file}, expected {mask_file}")
            continue

        output_image_name = compress_to_gzip(file)
        output_mask_name = compress_to_gzip(mask_file)

        preprocess_case(image_path, mask_path, os.path.join(output_image_dir, output_image_name), os.path.join(output_mask_dir, output_mask_name))
        print(f"Preprocessed {file}")

def preprocess_all(input_dir, output_dir):
    for split in ["test", "train", "val"]:
        preprocess_split(os.path.join(input_dir, split), os.path.join(output_dir, split))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    preprocess_all(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
