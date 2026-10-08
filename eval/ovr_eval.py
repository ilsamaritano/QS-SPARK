"""
OvR authentication evaluation.

Runs the trained OvR classifier on the genuine test split and computes
AUROC / EER / TAR@FAR=1% for satellite identity verification.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from data.dataset import build_ovr_datasets, make_loader
from models.hybrid import load_model_from_ckpt
from eval.metrics import auroc, eer, tar_at_far


@torch.no_grad()
def evaluate_ovr(target_sat: str,
                 ckpt_path: str,
                 all_satellites=None,
                 device: str = None,
                 far_target: float = 0.01) -> dict:
    """
    Evaluate OvR authentication on the genuine test split.

    Returns dict with keys:
        target_sat, backbone, fusion, auroc, eer, tar_at_far1pct, n_test
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = load_model_from_ckpt(ckpt, device)
    model.eval()

    sats = all_satellites or cfg.REDUCED_SATELLITES
    _, _, test_ds = build_ovr_datasets(target_sat, sats,
                                        seed=ckpt.get("seed", cfg.SPLIT_SEED),
                                        augment_train=False)

    loader = make_loader(test_ds, cfg.BATCH_SIZE, shuffle=False)

    all_labels, all_scores = [], []
    for imgs, labels in loader:
        out    = model(imgs.to(device))
        probs  = torch.softmax(out[0], dim=1)[:, 1].cpu().numpy()
        all_scores.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    y = np.array(all_labels)
    s = np.array(all_scores)

    return {
        "target_sat":    target_sat,
        "backbone":      ckpt["backbone"],
        "fusion":        ckpt.get("fusion"),
        "auroc":         auroc(y, s),
        "eer":           eer(y, s),
        "tar_at_far1pct": tar_at_far(y, s, far_target),
        "n_test":        len(test_ds),
    }
