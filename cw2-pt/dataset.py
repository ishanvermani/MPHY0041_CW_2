from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from preprocess import image_file_to_mask_file

class MalePelvicDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "masks"

        self.image_files = sorted(
            [p for p in self.image_dir.iterdir() if p.name.endswith(".nii") or p.name.endswith(".nii.gz")]
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

        mask_nib = np.rint(mask_nib).astype(np.int64)
        mask_nib = np.clip(mask_nib,0, 8)

        image_zyx = np.transpose(image_nib, (2, 1, 0))
        mask_zyx = np.transpose(mask_nib, (2, 1, 0))

        image = torch.from_numpy(image_zyx[None, ...]).float() # (C = 1, Z, Y, X)
        mask = torch.from_numpy(mask_zyx).long() # (Z, Y, X)

        return image, mask

class PatchDataset(Dataset):
    def __init__(self, vol_dataset: Dataset, patch_size =(16, 192, 192), foreground_prob=0.8,samples_per_volume=8):

        self.dataset = vol_dataset
        self.pz, self.py, self.px = patch_size
        self.foreground_prob = foreground_prob
        self.samples_per_volume = samples_per_volume

    def __len__(self):
        return len(self.dataset) * self.samples_per_volume

    def __pad__(self, image, mask):
        _, z, y, x = image.shape
        pad_z = max(0, self.pz - z)
        pad_y = max(0, self.py - y)
        pad_x = max(0, self.px - x)
        
        if pad_z or pad_y or pad_x:
            image = F.pad(image, (0, pad_x, 0, pad_y, 0, pad_z), value=0.0)
            mask = F.pad(mask, (0, pad_x, 0, pad_y, 0, pad_z), value=0)
        return image, mask
    
    def __getitem__(self, idx):
        vol_idx = idx // self.samples_per_volume
        image, mask = self.dataset[vol_idx]

        image, mask = self.__pad__(image, mask)
        _, z, y, x = image.shape

        if torch.rand(1).item() < self.foreground_prob and (mask > 0).any():
            fg = torch.nonzero(mask > 0, as_tuple=False)
            zc, yc, xc = fg[torch.randint(0, fg.shape[0], (1,)).item()].tolist()
            z0 = max(0, min(zc - self.pz // 2, z - self.pz))
            y0 = max(0, min(yc - self.py // 2, y - self.py))
            x0 = max(0, min(xc - self.px // 2, x - self.px))
        else:
            z0 = torch.randint(0, z - self.pz + 1, (1,)).item()
            y0 = torch.randint(0, y - self.py + 1, (1,)).item()
            x0 = torch.randint(0, x - self.px + 1, (1,)).item()

        image_patch = image[:, z0:z0+self.pz, y0:y0+self.py, x0:x0+self.px]
        mask_patch = mask[z0:z0+self.pz, y0:y0+self.py, x0:x0+self.px]

        return image_patch, mask_patch
