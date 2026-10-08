"""
Grad-CAM / Score-CAM explainability for the classical branch.

Identifies which spectrogram frequency–time regions drive the model's decision.
Uses Captum's LayerGradCam targeting the last convolutional layer of the backbone.

Output: per-image heatmap overlaid on the original spectrogram, saved as PNG.
For a batch, also computes the MEAN heatmap across the class (which frequency
bins are systematically important for genuine vs. replay discrimination).

Usage
-----
python3.10 explain/gradcam.py \\
    --ckpt results/ovr/Gaofen12/se_resnet34__sum/seed42/best.pt \\
    --target_sat Gaofen12 \\
    --label 0          # 0=genuine, 1=replay
    --n_samples 16
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import LayerGradCam, LayerAttribution
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from data.dataset import discover_images, SpectrogramDataset, make_loader, make_transform
from models.hybrid import load_model_from_ckpt


def load_model_and_layer(ckpt_path: str, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = load_model_from_ckpt(ckpt, device)
    model.eval()
    target_layer = model.get_last_conv()
    return model, target_layer, ckpt


def _forward_for_gradcam(model):
    """Wrapper: forward pass returning only the fused logits for Captum."""
    def forward_fn(x):
        return model(x)[0]   # fused_logits (or c_logits for classical-only)
    return forward_fn


def compute_gradcam(model, target_layer, imgs: torch.Tensor,
                    target_class: int = 1) -> np.ndarray:
    """
    Compute Grad-CAM heatmaps for a batch of images.
    Returns upsampled heatmaps of shape (B, 224, 224).
    """
    fwd = _forward_for_gradcam(model)
    lgc = LayerGradCam(fwd, target_layer)
    attr = lgc.attribute(imgs, target=target_class)
    # Upsample to input resolution
    upsampled = LayerAttribution.interpolate(attr, (224, 224), interpolate_mode="bilinear")
    # ReLU + normalise per image
    upsampled = upsampled.squeeze(1).detach().cpu().numpy()  # (B, 224, 224)
    upsampled = np.maximum(upsampled, 0)
    mx = upsampled.max(axis=(1, 2), keepdims=True) + 1e-8
    return upsampled / mx


def plot_gradcam_grid(orig_imgs: np.ndarray, heatmaps: np.ndarray,
                      title: str, out_path: str, n_cols: int = 4):
    n = min(len(orig_imgs), len(heatmaps))
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 4, n_rows * 2))
    axes = axes.flatten()

    for i in range(n):
        ax_img = axes[i * 2]
        ax_cam = axes[i * 2 + 1]
        ax_img.imshow(orig_imgs[i], cmap="gray", vmin=0, vmax=1)
        ax_img.axis("off")
        ax_cam.imshow(orig_imgs[i], cmap="gray", vmin=0, vmax=1)
        ax_cam.imshow(heatmaps[i], cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax_cam.axis("off")

    for j in range(n * 2, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  [Grad-CAM] Saved {out_path}")


def run_gradcam(ckpt_path: str, target_sat: str, label: int = 1,
                n_samples: int = 16, out_dir: str = None, device: str = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model, target_layer, ckpt = load_model_and_layer(ckpt_path, device)
    sats = ckpt.get("all_sats", cfg.REDUCED_SATELLITES)

    label_name = "genuine" if label == 0 else "replay"
    all_recs   = discover_images(cfg.SPECTROGRAM_ROOT, sats)
    records    = [r for r in all_recs
                  if r["satellite"] == target_sat and r["label_name"] == label_name][:n_samples]

    if not records:
        print(f"  No {label_name} images for {target_sat}")
        return

    transform = make_transform(augment=False)
    ds        = SpectrogramDataset(records, transform=transform)
    loader    = make_loader(ds, batch_size=min(n_samples, 8), shuffle=False, num_workers=0)

    all_imgs, all_heatmaps = [], []
    for imgs, _ in loader:
        imgs_dev = imgs.to(device).requires_grad_(True)
        hmaps    = compute_gradcam(model, target_layer, imgs_dev, target_class=label)
        all_heatmaps.append(hmaps)
        # Convert tensor to numpy [0,1] for display
        raw = imgs.squeeze(1).cpu().numpy() * 0.5 + 0.5
        all_imgs.append(raw)

    all_imgs     = np.concatenate(all_imgs, axis=0)
    all_heatmaps = np.concatenate(all_heatmaps, axis=0)

    out_dir = Path(out_dir or (Path(cfg.RESULTS_ROOT) / "explain"))
    out_dir.mkdir(parents=True, exist_ok=True)
    mkey    = f"{ckpt['backbone']}__{ckpt.get('fusion', 'classical')}"
    out_path = out_dir / f"gradcam_{target_sat}_{label_name}_{mkey}.png"

    plot_gradcam_grid(
        all_imgs, all_heatmaps,
        title=f"Grad-CAM: {target_sat} [{label_name}]  model={mkey}",
        out_path=str(out_path),
    )

    # Mean heatmap across samples (which frequency bands are consistently important)
    mean_hmap = all_heatmaps.mean(axis=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(all_imgs.mean(axis=0), cmap="gray")
    ax1.set_title(f"Mean spectrogram ({label_name})")
    ax1.axis("off")
    ax2.imshow(all_imgs.mean(axis=0), cmap="gray")
    ax2.imshow(mean_hmap, cmap="jet", alpha=0.6)
    ax2.set_title(f"Mean Grad-CAM (class={label_name})")
    ax2.axis("off")
    plt.suptitle(f"{target_sat} [{mkey}]", fontsize=10)
    plt.tight_layout()
    mean_path = out_dir / f"gradcam_mean_{target_sat}_{label_name}_{mkey}.png"
    plt.savefig(mean_path, dpi=150)
    plt.close()
    print(f"  [Grad-CAM] Mean heatmap -> {mean_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       required=True)
    ap.add_argument("--target_sat", required=True)
    ap.add_argument("--label",      type=int, default=1, help="0=genuine 1=replay")
    ap.add_argument("--n_samples",  type=int, default=16)
    ap.add_argument("--out_dir",    default=None)
    ap.add_argument("--device",     default=None)
    a = ap.parse_args()
    run_gradcam(a.ckpt, a.target_sat, a.label, a.n_samples, a.out_dir, a.device)
