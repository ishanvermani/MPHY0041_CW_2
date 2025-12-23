from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing import normalize_image, sanitize_mask

def make_pairs(split_dir: str) -> list[tuple[str, str]]:
    split_dir = Path(split_dir)
    img_dir = split_dir / "images"
    msk_dir = split_dir / "masks"

    img_files = sorted(list(img_dir.glob("*.nii")) + list(img_dir.glob("*.nii.gz")))
    if not img_files:
        raise RuntimeError(f"No images found in {img_dir}")

    pairs = []
    for ip in img_files:
        mp = msk_dir / ip.name.replace("_img.nii.gz", "_mask.nii.gz").replace("_img.nii", "_mask.nii")
        if not mp.exists():
            raise FileNotFoundError(f"Missing mask for {ip.name}: expected {mp}")
        pairs.append((str(ip), str(mp)))
    return pairs

class Dataset(Dataset):
    def __init__(self, split_dir: str, normalize: str = "zscore"):
        self.pairs = make_pairs(split_dir)
        self.normalize = normalize

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = nib.load(img_path).get_fdata.astype(np.float32)
        mask = nib.load(mask_path).get_fdata

        img = normalize_image(img)
        mask = sanitize_mask(mask)

        img = img[None. ...]

        return {
            "image": torch.from_numpy(img).float(),
            "mask": torch.from_numpy(mask).long(),
            "id": Path(img_path).stem
        }