import numpy as np

def normalize_image(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    img = img.astype(np.float32)

    nz = img != 0
    if nz.sum() > 10:
        mean = float(img[nz].mean())
        std = float(img[nz].std())
        return (img - mean) / (std + eps)
    return img

def sanitize_mask(mask: np.ndarray, num_classes: int = 9) -> np.ndarray:
    mask = np.rint(mask).astype(np.int16)
    mn, mx = int(mask.min()), int(mask.max())
    if mn < 0 or mx >= num_classes:
        raise ValueError(f"Mask labels out of range: {mn}..{mx} (expected 0..{num_classes-1})")
    return mask
