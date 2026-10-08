"""
Adaptive white-box PGD replay attack against the Mahalanobis detector.

Matches PLASyR's adaptive_replay_sweep.py protocol:
  - PGD in IMAGE DOMAIN, L∞-bounded, minimising the Mahalanobis score.
  - Threshold θ fixed from genuine FAR=1% (deployed detector scenario).
  - ε-sweep: reports TRR@FAR=1% degradation curve.
  - Backpropagation flows through the full model (classical + VQC if hybrid).

For hybrid models, minimising Mahalanobis in the JOINT embedding space
requires gradients through both the backbone and the VQC (default.qubit +
backprop diff_method supports this).

Outputs
-------
results/adaptive_pgd/<model_key>/<satellite>/sweep.json
results/adaptive_pgd/macro_curve.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from data.dataset import (discover_images, SpectrogramDataset,
                           make_loader, make_transform, get_split_records)
from models.hybrid import model_key, load_model_from_ckpt
from eval.metrics import GaussianMahalanobis, auroc, eer, trr_at_far

VALID_MIN, VALID_MAX = -1.0, 1.0


# ── Differentiable Mahalanobis (torch) ────────────────────────────────────────

def _gm_to_torch(gm: GaussianMahalanobis, device):
    """
    Mirror GaussianMahalanobis.score() as differentiable torch ops. When the
    Gaussian was fit with PCA (train N <= embedding dim, the common case here
    with ~70-160 genuine images vs 512-1024d features), score() projects the
    standardised embedding through the PCA basis before computing Mahalanobis
    distance in the reduced space — this must be replicated here or the
    dimensions of mu/inv_cov (n_components) won't match the raw embedding (D).
    """
    out = {
        "mu_f":  torch.tensor(gm.mu_f,    dtype=torch.float32, device=device),
        "sd_f":  torch.tensor(gm.sd_f,    dtype=torch.float32, device=device),
        "mu":    torch.tensor(gm.mu,       dtype=torch.float32, device=device),
        "P":     torch.tensor(gm.inv_cov,  dtype=torch.float32, device=device),
        "pca_mean": None,
        "pca_components": None,
    }
    if gm.pca is not None:
        out["pca_mean"]       = torch.tensor(gm.pca.mean_,       dtype=torch.float32, device=device)
        out["pca_components"] = torch.tensor(gm.pca.components_, dtype=torch.float32, device=device)
    return out


def _maha_torch(z: torch.Tensor, gmt: dict) -> torch.Tensor:
    zn = (z - gmt["mu_f"]) / gmt["sd_f"]
    if gmt["pca_mean"] is not None:
        zn = (zn - gmt["pca_mean"]) @ gmt["pca_components"].T
    d  = zn - gmt["mu"]
    return torch.einsum("bi,ij,bj->b", d, gmt["P"], d)


# ── PGD minimiser ─────────────────────────────────────────────────────────────

def pgd_min_maha(model, x: torch.Tensor, eps: float, steps: int,
                 gmt: dict, device) -> torch.Tensor:
    """PGD that MINIMISES Mahalanobis distance (drives replay toward genuine manifold)."""
    if eps == 0:
        return x.detach()

    alpha = 2.5 * eps / steps   # Madry-style step size
    x0    = x.detach()
    xa    = x0.clone()

    for _ in range(steps):
        xa = xa.requires_grad_(True)
        emb  = model.get_joint_embedding(xa)
        loss = _maha_torch(emb, gmt).sum()
        grad = torch.autograd.grad(loss, xa)[0]
        xa   = (xa.detach() - alpha * grad.sign())
        xa   = torch.clamp(xa, x0 - eps, x0 + eps).clamp(VALID_MIN, VALID_MAX).detach()

    return xa


# ── Per-satellite sweep ───────────────────────────────────────────────────────

@torch.no_grad()
def _embed_all(model, records, device):
    if not records:
        return np.empty((0, model.feature_dim))
    ds     = SpectrogramDataset(records, transform=make_transform(augment=False))
    loader = make_loader(ds, cfg.BATCH_SIZE, shuffle=False)
    model.eval()
    return np.concatenate(
        [model.get_joint_embedding(imgs.to(device)).cpu().numpy() for imgs, _ in loader]
    )


def sweep_satellite(target_sat: str, ckpt_path: str, all_satellites,
                    epsilons, pgd_steps: int, device, far_target: float = 0.01,
                    max_attack: int = 500) -> dict:
    """Run PGD ε-sweep for one satellite. Returns per-ε metrics."""
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = load_model_from_ckpt(ckpt, device)

    sats = all_satellites or cfg.REDUCED_SATELLITES

    # Genuine train (fit Gaussian) and test (calibrate θ)
    train_recs, _, test_recs = get_split_records(
        target_sat, sats, seed=ckpt.get("seed", cfg.SPLIT_SEED)
    )
    all_recs    = discover_images(cfg.SPECTROGRAM_ROOT, sats)
    replay_recs = [r for r in all_recs
                   if r["label_name"] == "replay" and r["satellite"] == target_sat]

    if not train_recs or not replay_recs:
        return {}

    # Cap replay images attacked (PGD is expensive)
    if max_attack and len(replay_recs) > max_attack:
        step = max(1, len(replay_recs) // max_attack)
        replay_recs = replay_recs[::step][:max_attack]

    g_train_emb = _embed_all(model, train_recs, device)
    g_test_emb  = _embed_all(model, test_recs,  device)

    gm  = GaussianMahalanobis().fit(g_train_emb)
    gmt = _gm_to_torch(gm, device)
    g_scores = gm.score(g_test_emb)
    theta    = float(np.percentile(g_scores, 100.0 * (1.0 - far_target)))

    # Pre-load replay images as tensors
    replay_ds     = SpectrogramDataset(replay_recs, transform=make_transform(augment=False))
    replay_loader = make_loader(replay_ds, min(16, cfg.BATCH_SIZE),
                                 shuffle=False, num_workers=0)

    eps_results = {}
    for eps in epsilons:
        print(f"    eps={eps:.3f} ...", end="", flush=True)
        r_scores = []
        for imgs, _ in replay_loader:
            imgs = imgs.to(device)
            xa   = pgd_min_maha(model, imgs, eps, pgd_steps, gmt, device)
            with torch.no_grad():
                emb = model.get_joint_embedding(xa)
            r_scores.extend(gm.score(emb.cpu().numpy()).tolist())

        r_scores_np = np.array(r_scores)
        trr, _      = trr_at_far(g_scores, r_scores_np, far_target)
        y           = np.concatenate([np.zeros(len(g_scores)), np.ones(len(r_scores_np))])
        all_s       = np.concatenate([g_scores, r_scores_np])
        eps_results[eps] = {
            "trr":   trr,
            "auroc": auroc(y, all_s),
            "eer":   eer(y, all_s),
        }
        print(f" TRR={trr:.3f} AUROC={eps_results[eps]['auroc']:.3f}")

    return eps_results


# ── Entry point ───────────────────────────────────────────────────────────────

def run_adaptive_sweep(backbones=None, fusions=None, satellites=None,
                        seeds=None, epsilons=None, pgd_steps: int = 50,
                        device: str = None, out_root: str = None,
                        ckpt_root: str = None, max_attack: int = 500):
    """
    ckpt_root: directory to read checkpoints from (defaults to out_root).
    Kept separate from out_root so multiple parallel processes (e.g. one per
    GPU, each covering a different backbone) can read from the same shared
    checkpoint tree while writing their own results to distinct out_roots
    without racing on the same CSV file.
    """
    device   = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backbones  = backbones  or cfg.BACKBONES
    fusions    = (["classical"] + cfg.FUSION_OPS) if fusions is None else fusions
    satellites = satellites or cfg.REDUCED_SATELLITES
    seeds      = seeds      or [cfg.SEED]
    epsilons   = epsilons   or [0, 0.02, 0.05, 0.10, 0.20, 0.30]
    out_root   = Path(out_root or cfg.RESULTS_ROOT)
    ckpt_root  = Path(ckpt_root or out_root)

    all_rows = []
    for backbone in backbones:
        for fusion_arg in fusions:
            fusion_name = None if fusion_arg == "classical" else fusion_arg
            mkey = model_key(backbone, fusion_name)

            for seed in seeds:
                for sat in satellites:
                    ckpt_path = (ckpt_root / "ovr" / sat / mkey / f"seed{seed}" / "best.pt")
                    if not ckpt_path.exists():
                        continue
                    print(f"\n[PGD] {sat}  [{mkey}]  seed={seed}")
                    eps_res = sweep_satellite(
                        target_sat=sat, ckpt_path=str(ckpt_path),
                        all_satellites=satellites, epsilons=epsilons,
                        pgd_steps=pgd_steps, device=device,
                        max_attack=max_attack,
                    )
                    for eps, metrics in eps_res.items():
                        all_rows.append({
                            "backbone": backbone, "fusion": fusion_arg,
                            "satellite": sat, "seed": seed,
                            "eps": eps, **metrics,
                        })

    if not all_rows:
        print("No results.")
        return

    import pandas as pd
    df = pd.DataFrame(all_rows)

    # Macro curve: mean/std across satellites and seeds per (backbone, fusion, eps)
    macro = df.groupby(["backbone", "fusion", "eps"]).agg(
        trr_mean=("trr",   "mean"), trr_std=("trr",   "std"),
        auroc_mean=("auroc","mean"), auroc_std=("auroc","std"),
    ).reset_index()

    pgd_dir = out_root / "adaptive_pgd"
    pgd_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(pgd_dir / "per_sat_sweep.csv", index=False)
    macro.to_csv(pgd_dir / "macro_curve.csv", index=False)
    with open(pgd_dir / "all_rows.json", "w") as f:
        json.dump(all_rows, f, indent=2)

    print(f"\nAdaptive PGD sweep saved to {pgd_dir}")
    return df, macro


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones",   nargs="+", default=cfg.BACKBONES)
    ap.add_argument("--fusions",     nargs="+", default=["classical"] + cfg.FUSION_OPS)
    ap.add_argument("--satellites",  nargs="+", default=cfg.REDUCED_SATELLITES)
    ap.add_argument("--seeds",       nargs="+", type=int, default=[cfg.SEED])
    ap.add_argument("--epsilons",    nargs="+", type=float,
                    default=[0, 0.02, 0.05, 0.10, 0.20, 0.30])
    ap.add_argument("--pgd_steps",   type=int, default=50)
    ap.add_argument("--max_attack",  type=int, default=500)
    ap.add_argument("--device",      default=None)
    ap.add_argument("--out_root",    default=cfg.RESULTS_ROOT)
    ap.add_argument("--ckpt_root",   default=None)
    a = ap.parse_args()
    run_adaptive_sweep(
        backbones=a.backbones, fusions=a.fusions, satellites=a.satellites,
        seeds=a.seeds, epsilons=a.epsilons, pgd_steps=a.pgd_steps,
        device=a.device, out_root=a.out_root, ckpt_root=a.ckpt_root,
        max_attack=a.max_attack,
    )
