from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess import image_file_to_mask_file

class MalePelvicDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "masks"

        self.image_files = sorted(
            [p for p in self.image_dir.iterdir() if p.endswith(".nii") or p.endswith(".nii.gz")]
        )

        # No need to normalize because done already in preprocess.py

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_name = image_file_to_mask_file(image_path.name)
        mask_path = self.mask_dir / mask_name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {image_path.name}, expected {mask_name}")

        image_nib = nib.load(str(image_path)).get_fdata(dtype=np.float32)
        mask_nib = nib.load(str(mask_path)).get_fdata(dtype=np.float32)

        mask = np.rint(mask_nib).astype(np.int64)
        mask = np.clip(mask_nib,0, 8)

        image_zyx = np.transpose(image_nib, (2, 1, 0))
        mask_zyx = np.transpose(mask_nib, (2, 1, 0))

        image = torch.from_numpy(image_zyx[None, ...]).float() # (C = 1, Z, Y, X)
        mask = torch.from_numpy(mask_zyx).long() # (Z, Y, X)

        return image, mask

