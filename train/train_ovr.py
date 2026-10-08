"""
OvR (One-vs-Rest) satellite authentication trainer.

Follows PLASyR training protocol:
- Only genuine images are used during training.
- Binary classifier: target satellite (label=1) vs all others (label=0).
- The resulting embedding space is used for Mahalanobis-based replay detection.

Supports both ClassicalOnlyModel (fusion_name=None) and HybridModel.
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from data.dataset import build_ovr_datasets, make_loader, get_class_weights
from models.hybrid import build_model, model_key, HybridModel


# ── Training / evaluation loops ───────────────────────────────────────────────

def _loss_fn(model, logits, labels, class_weights):
    """
    ProductFusion outputs log-probs (NLLLoss); all other fusions use CrossEntropyLoss.
    Detects the fusion type from the model's fusion attribute.
    """
    fusion = getattr(model, "fusion", None)
    is_log_prob = getattr(fusion, "is_log_prob", False)
    if is_log_prob:
        return F.nll_loss(logits, labels, weight=class_weights)
    return F.cross_entropy(logits, labels, weight=class_weights)


def _run_epoch(model, loader, device, class_weights,
               optimizer=None, grad_clip_norm: float = None) -> tuple:
    """Single train or eval epoch. Returns (avg_loss, auroc)."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_labels, all_scores = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            cw     = class_weights.to(device)

            out    = model(imgs)
            logits = out[0]
            loss   = _loss_fn(model, logits, labels, cw)

            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
            all_scores.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except Exception:
        auc = 0.5
    return avg_loss, auc


def _build_optimizer(model, lr: float, quantum_lr: float, weight_decay: float):
    """
    Separate parameter groups for classical vs. quantum branch. VQC rotation-gate
    gradients are small relative to the pretrained CNN backbone's, so a single
    shared LR under-trains the quantum branch within the epoch budget; giving
    it its own (higher) LR lets it actually converge. No-op for classical-only
    models (falls back to a single group at `lr`).
    """
    if not isinstance(model, HybridModel) or quantum_lr is None:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    quantum_params, classical_params = [], []
    for name, p in model.named_parameters():
        if name.startswith("vqc_block") or name.startswith("q_head"):
            quantum_params.append(p)
        else:
            classical_params.append(p)

    return torch.optim.Adam([
        {"params": classical_params, "lr": lr},
        {"params": quantum_params,   "lr": quantum_lr},
    ], weight_decay=weight_decay)


# ── Public API ────────────────────────────────────────────────────────────────

def train_ovr(target_sat: str,
              backbone_name: str,
              fusion_name: str = None,
              all_satellites=None,
              seed: int = cfg.SEED,
              epochs: int = cfg.EPOCHS_REDUCED,
              device: str = None,
              out_root: str = None,
              n_qubits: int = cfg.N_QUBITS,
              n_layers: int = cfg.VQC_LAYERS,
              quantum_lr: float = None,
              grad_clip_norm: float = None) -> dict:
    """
    Train one OvR classifier and save the best checkpoint.

    Parameters
    ----------
    target_sat    : satellite to identify (positive class)
    backbone_name : "se_resnet34" | "densenet121"
    fusion_name   : fusion op name, or None for classical-only
    all_satellites: list of satellites included as negatives; defaults to cfg.REDUCED_SATELLITES
    seed          : used for both data split and torch manual seed
    epochs        : training epochs
    device        : torch device string; defaults to cuda if available
    out_root      : root directory for checkpoints; defaults to cfg.RESULTS_ROOT
    n_qubits      : VQC width (hybrid models only)
    n_layers      : VQC depth (hybrid models only)
    quantum_lr    : separate LR for vqc_block/q_head params; None uses a single
                    shared optimizer at cfg.LR (matches original behavior)
    grad_clip_norm: if set, clips gradient norm before each optimizer step
                    (recommended when quantum_lr > cfg.LR, to keep the higher
                    LR from destabilizing the pretrained classical backbone)

    Returns
    -------
    dict with keys: target_sat, backbone, fusion, seed, best_val_auroc, test_auroc,
                    ckpt_path, elapsed_s, history
    """
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sats   = all_satellites or cfg.REDUCED_SATELLITES

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = build_ovr_datasets(target_sat, sats, seed=seed)
    if len(train_ds) == 0:
        return {"error": f"No training data for {target_sat}"}

    class_weights = get_class_weights(train_ds)
    train_loader  = make_loader(train_ds, cfg.BATCH_SIZE, shuffle=True)
    val_loader    = make_loader(val_ds,   cfg.BATCH_SIZE, shuffle=False)
    test_loader   = make_loader(test_ds,  cfg.BATCH_SIZE, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(backbone_name, fusion_name, num_classes=2,
                        n_qubits=n_qubits, n_layers=n_layers).to(device)

    optimizer = _build_optimizer(model, cfg.LR, quantum_lr, cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # ── Output paths ──────────────────────────────────────────────────────────
    mkey      = model_key(backbone_name, fusion_name)
    ckpt_dir  = Path(out_root or cfg.RESULTS_ROOT) / "ovr" / target_sat / mkey / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auroc = 0.0
    history        = []
    t0             = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_auc = _run_epoch(model, train_loader, device, class_weights,
                                     optimizer, grad_clip_norm=grad_clip_norm)
        va_loss, va_auc = _run_epoch(model, val_loader,   device, class_weights)
        scheduler.step()

        history.append({"epoch": epoch,
                         "tr_loss": tr_loss, "tr_auroc": tr_auc,
                         "va_loss": va_loss, "va_auroc": va_auc})

        if va_auc > best_val_auroc:
            best_val_auroc = va_auc
            torch.save({
                "model_state":  model.state_dict(),
                "backbone":     backbone_name,
                "fusion":       fusion_name,
                "target_sat":   target_sat,
                "seed":         seed,
                "epoch":        epoch,
                "all_sats":     sats,
                "n_qubits":     n_qubits,
                "n_layers":     n_layers,
            }, ckpt_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{epoch:3d}/{epochs}] "
                  f"tr_loss={tr_loss:.4f} tr_auc={tr_auc:.4f} | "
                  f"va_loss={va_loss:.4f} va_auc={va_auc:.4f}")

    # ── Test evaluation with best model ───────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    te_loss, te_auc = _run_epoch(model, test_loader, device, class_weights)

    elapsed = time.time() - t0
    results = {
        "target_sat":     target_sat,
        "backbone":       backbone_name,
        "fusion":         fusion_name,
        "seed":           seed,
        "best_val_auroc": best_val_auroc,
        "test_auroc":     te_auc,
        "train_n":        len(train_ds),
        "val_n":          len(val_ds),
        "test_n":         len(test_ds),
        "elapsed_s":      elapsed,
        "ckpt_path":      str(ckpt_path),
        "history":        history,
    }

    with open(ckpt_dir / "train_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  → {target_sat} [{mkey}] seed={seed} "
          f"test_auroc={te_auc:.4f}  ({elapsed:.0f}s)")
    return results
