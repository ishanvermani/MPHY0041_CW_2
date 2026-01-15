import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt

from src.model import UNet

import torch.nn.functional as F


def load_img_as_zyx(image_nii_path: str):
    """
    Loads a preprocessed nifit image and returns:
      - nib object (for affine/header)
      - image_zyx float32 with shape [Z,Y,X]
    """
    img_nib = nib.load(image_nii_path)
    img_xyz = img_nib.get_fdata(dtype=np.float32)
    img_zyx = np.transpose(img_xyz, (2, 1, 0))
    return img_nib, img_zyx


def load_mask_as_zyx(mask_nii_path: str):
    """ Loads a mask and returns mask_zyx int64 [Z,Y,X] with clipping. """
    mask_nib = nib.load(mask_nii_path)
    mask_xyz = mask_nib.get_fdata(dtype=np.float32)
    mask_xyz = np.rint(mask_xyz).astype(np.int64)
    mask_xyz = np.clip(mask_xyz, 0, 8)
    mask_zyx = np.transpose(mask_xyz, (2, 1, 0))
    return mask_nib, mask_zyx

def pad_to_multiple(x, multiple=16):
    """ Pads a 4D tensor so height and width are divisible by the given multiple. """
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    # pad on right/bottom
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x_pad, (h, w)

@torch.no_grad()
def predict_volume_zyx(model, img_zyx, device, batch_slices=8, multiple=16):
    """ Runs slice-wise model inference on a 3D volume and returns predictions in [Z,Y,X] format. """
    model.eval()
    Z, Y, X = img_zyx.shape
    pred_zyx = np.zeros((Z, Y, X), dtype=np.uint8)

    z = 0
    while z < Z:
        z2 = min(Z, z + batch_slices)
        x = torch.from_numpy(img_zyx[z:z2]).unsqueeze(1).to(device)

        x_pad, (orig_h, orig_w) = pad_to_multiple(x, multiple=multiple)
        logits = model(x_pad)
        logits = logits[:, :, :orig_h, :orig_w]

        preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
        pred_zyx[z:z2] = preds
        z = z2

    return pred_zyx


def save_pred_like_image(image_nib: nib.Nifti1Image, pred_zyx: np.ndarray, out_path: str):
    """ Convert prediction from [Z,Y,X] back to [X,Y,Z] and save with same affine/header as the image. """
    pred_xyz = np.transpose(pred_zyx, (2, 1, 0)).astype(np.uint8)
    out = nib.Nifti1Image(pred_xyz, image_nib.affine, image_nib.header)
    nib.save(out, out_path)


def overlay_pngs(img_zyx, pred_flat_zyx, pred_hier_zyx, gt_zyx, out_dir: Path, slices, alpha=0.45):
    """ Saves PNG overlays of image slices with flat, hierarchical, and GT segmentations. """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab20")

    for z in slices:
        if z < 0 or z >= img_zyx.shape[0]:
            continue

        img = img_zyx[z]
        pf = pred_flat_zyx[z]
        ph = pred_hier_zyx[z]
        gt = gt_zyx[z] if gt_zyx is not None else None

        ncols = 4 if gt_zyx is not None else 3
        fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5))

        axes[0].imshow(img, cmap="gray")
        axes[0].set_title(f"Image (z={z})")
        axes[0].axis("off")

        axes[1].imshow(img, cmap="gray")
        axes[1].imshow(pf, cmap=cmap, alpha=alpha, vmin=0, vmax=max(1, int(pf.max())))
        axes[1].set_title("Flat prediction")
        axes[1].axis("off")

        axes[2].imshow(img, cmap="gray")
        axes[2].imshow(ph, cmap=cmap, alpha=alpha, vmin=0, vmax=max(1, int(ph.max())))
        axes[2].set_title("Hier prediction")
        axes[2].axis("off")

        if gt_zyx is not None:
            axes[3].imshow(img, cmap="gray")
            axes[3].imshow(gt, cmap=cmap, alpha=alpha, vmin=0, vmax=max(1, int(gt.max())))
            axes[3].set_title("Ground truth")
            axes[3].axis("off")

        fig.savefig(out_dir / f"overlay_z{z:03d}.png", dpi=150)
        plt.close(fig)


def parse_slices(spec: str, Z: int):
     """ Parses a slice specification string into a list of valid Z indices. """
    spec = spec.strip().lower()
    if spec == "mid":
        return [Z // 2]
    if spec == "all":
        return list(range(Z))
    if ":" in spec:
        parts = [p for p in spec.split(":") if p != ""]
        if len(parts) == 2:
            a, b = map(int, parts)
            step = 1
        elif len(parts) == 3:
            a, b, step = map(int, parts)
        else:
            print("Slice range is wrong!")
        return list(range(a, min(b, Z), step))
    return [int(x) for x in spec.split(",") if x.strip() != ""]

def load_checkpoint(model, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_nii", required=True)
    ap.add_argument("--flat_ckpt", required=True)
    ap.add_argument("--hier_ckpt", required=True)
    ap.add_argument("--num_classes", type=int, default=9)

    ap.add_argument("--mask_nii", default=None)
    ap.add_argument("--out_dir", default="prediction_masks")
    ap.add_argument("--batch_slices", type=int, default=8)
    ap.add_argument("--slices", default="mid")
    ap.add_argument("--alpha", type=float, default=0.45)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_nib, img_zyx = load_img_as_zyx(args.image_nii)

    gt_zyx = None
    if args.mask_nii is not None:
        _, gt_zyx = load_mask_as_zyx(args.mask_nii)

    flat_model = load_checkpoint(UNet(in_channels=1, num_classes=args.num_classes), args.flat_ckpt, device)
    hier_model = load_checkpoint(UNet(in_channels=1, num_classes=args.num_classes), args.hier_ckpt, device)

    pred_flat_zyx = predict_volume_zyx(flat_model, img_zyx, device, batch_slices=args.batch_slices)
    pred_hier_zyx = predict_volume_zyx(hier_model, img_zyx, device, batch_slices=args.batch_slices)

    save_pred_like_image(img_nib, pred_flat_zyx, str(out_dir / "pred_flat.nii.gz"))
    save_pred_like_image(img_nib, pred_hier_zyx, str(out_dir / "pred_hier.nii.gz"))

    slices = parse_slices(args.slices, Z=img_zyx.shape[0])
    overlay_pngs(img_zyx, pred_flat_zyx, pred_hier_zyx, gt_zyx, out_dir / "png_overlays", slices, alpha=args.alpha)

    print("Saved outputs:")
    print(f"  {out_dir / 'pred_flat.nii.gz'}")
    print(f"  {out_dir / 'pred_hier.nii.gz'}")
    print(f"  {out_dir / 'png_overlays'}")

if __name__ == "__main__":
    main()