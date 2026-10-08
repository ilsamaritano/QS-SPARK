"""
Mahalanobis-based replay detection evaluation.

Protocol (matching PLASyR replay_eval.py):
1. Load trained OvR model from checkpoint.
2. Use the same train/test split (reconstructed from checkpoint seed) to get
   genuine-train and genuine-test images for the target satellite.
3. Fit a GaussianMahalanobis on joint [classical, quantum] embeddings from
   the genuine TRAIN set.
4. Score genuine TEST images and ALL replay images of the target satellite.
5. Report AUROC / EER / TRR@FAR=1%.

For classical-only models, the joint embedding is just the classical features.
For hybrid models, it is cat([classical_features, quantum_features]).

Also reports classical-only and quantum-only Mahalanobis scores as ablations.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from data.dataset import (discover_images, SpectrogramDataset,
                           make_loader, make_transform, get_split_records)
from models.hybrid import load_model_from_ckpt
from eval.metrics import GaussianMahalanobis, auroc, eer, trr_at_far


@torch.no_grad()
def _embed(model, records, device):
    """Extract joint embeddings from a list of records."""
    if not records:
        return np.empty((0, model.feature_dim))
    ds     = SpectrogramDataset(records, transform=make_transform(augment=False))
    loader = make_loader(ds, cfg.BATCH_SIZE, shuffle=False)
    model.eval()
    embs = []
    for imgs, _ in loader:
        embs.append(model.get_joint_embedding(imgs.to(device)).cpu().numpy())
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def _embed_branches(model, records, device):
    """Extract (classical_feats, quantum_feats) separately for ablation."""
    if not records:
        feat_dim = getattr(model, "backbone", model).feature_dim
        return np.empty((0, feat_dim)), None
    ds     = SpectrogramDataset(records, transform=make_transform(augment=False))
    loader = make_loader(ds, cfg.BATCH_SIZE, shuffle=False)
    model.eval()
    c_all, q_all = [], []
    for imgs, _ in loader:
        out = model(imgs.to(device))
        # out = (fused_logits, c_feats, q_feats, c_logits, q_logits)
        c_all.append(out[1].cpu().numpy())
        if out[2] is not None:
            q_all.append(out[2].cpu().numpy())
    c_emb = np.concatenate(c_all, axis=0)
    q_emb = np.concatenate(q_all, axis=0) if q_all else None
    return c_emb, q_emb


def evaluate_replay_detection(target_sat: str,
                               ckpt_path: str,
                               all_satellites=None,
                               device: str = None,
                               far_target: float = 0.01) -> dict:
    """
    Load model, extract embeddings, fit Gaussian, score genuine + replay, report metrics.

    Returns a dict with keys:
        target_sat, backbone, fusion,
        auroc, eer, trr_at_far1pct, threshold,
        auroc_classical, eer_classical, trr_classical,   # classical-branch ablation
        auroc_quantum,   eer_quantum,   trr_quantum,     # quantum-branch ablation (None if classical)
        n_genuine_train, n_genuine_test, n_replay_test
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = load_model_from_ckpt(ckpt, device)
    model.eval()

    sats = all_satellites or cfg.REDUCED_SATELLITES

    # ── Split genuine images into train / test (same seed as training) ────────
    train_recs, _, test_recs = get_split_records(
        target_sat, sats, seed=ckpt.get("seed", cfg.SPLIT_SEED)
    )

    # ── Collect all replay images for target satellite ─────────────────────────
    all_records = discover_images(cfg.SPECTROGRAM_ROOT, sats)
    replay_recs = [r for r in all_records
                   if r["label_name"] == "replay" and r["satellite"] == target_sat]

    if not train_recs or not replay_recs:
        return {"error": f"Missing data for {target_sat}",
                "target_sat": target_sat, "backbone": ckpt["backbone"],
                "fusion": ckpt.get("fusion")}

    # ── Extract embeddings ────────────────────────────────────────────────────
    g_train_c, g_train_q = _embed_branches(model, train_recs, device)
    g_test_c,  g_test_q  = _embed_branches(model, test_recs,  device)
    r_c,       r_q       = _embed_branches(model, replay_recs, device)

    def _maha_metrics(g_tr, g_te, rep):
        if g_tr is None or len(g_tr) == 0:
            return None, None, None, None
        gm = GaussianMahalanobis().fit(g_tr)
        g_scores = gm.score(g_te)
        r_scores = gm.score(rep)
        y        = np.concatenate([np.zeros(len(g_scores)), np.ones(len(r_scores))])
        all_s    = np.concatenate([g_scores, r_scores])
        trr, theta = trr_at_far(g_scores, r_scores, far_target)
        return auroc(y, all_s), eer(y, all_s), trr, theta

    # Joint (classical + quantum) embeddings
    g_train_emb = _embed(model, train_recs,  device)
    g_test_emb  = _embed(model, test_recs,   device)
    r_emb       = _embed(model, replay_recs, device)

    joint_auc, joint_eer, joint_trr, joint_theta = _maha_metrics(
        g_train_emb, g_test_emb, r_emb
    )

    # Classical-branch ablation
    c_auc, c_eer, c_trr, _ = _maha_metrics(g_train_c, g_test_c, r_c)

    # Quantum-branch ablation (None for classical-only model)
    q_auc = q_eer = q_trr = None
    if g_train_q is not None:
        q_auc, q_eer, q_trr, _ = _maha_metrics(g_train_q, g_test_q, r_q)

    return {
        "target_sat":        target_sat,
        "backbone":          ckpt["backbone"],
        "fusion":            ckpt.get("fusion"),
        # Joint (primary metric)
        "auroc":             joint_auc,
        "eer":               joint_eer,
        "trr_at_far1pct":    joint_trr,
        "threshold":         joint_theta,
        # Ablations
        "auroc_classical":   c_auc,
        "eer_classical":     c_eer,
        "trr_classical":     c_trr,
        "auroc_quantum":     q_auc,
        "eer_quantum":       q_eer,
        "trr_quantum":       q_trr,
        "n_genuine_train":   len(train_recs),
        "n_genuine_test":    len(test_recs),
        "n_replay_test":     len(replay_recs),
    }
