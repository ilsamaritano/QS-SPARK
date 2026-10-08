"""
Quantum branch explainability.

Three analyses:

1. Per-qubit ⟨Z⟩ attribution
   Which qubits most distinguish genuine vs. replay?
   Compare mean |⟨Z_i⟩| across genuine and replay images.

2. Parameter-shift feature importance
   For each backbone feature dimension j, compute ∂⟨Z_total⟩/∂angle_j using
   the parameter-shift rule applied to the VQC INPUT angles (not VQC weights).
   This tells us which backbone features the quantum branch amplifies.

3. Gate activity report (GatedFusion only)
   For each sample, print the learned gate g(x) to show when the quantum
   branch is up-weighted vs. suppressed.

Usage
-----
python3.10 explain/quantum_attr.py \\
    --ckpt results/ovr/Gaofen12/se_resnet34__gated/seed42/best.pt \\
    --target_sat Gaofen12
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from data.dataset import discover_images, SpectrogramDataset, make_loader, make_transform
from models.hybrid import HybridModel, load_model_from_ckpt


# ── 1. Per-qubit ⟨Z⟩ means ───────────────────────────────────────────────────

@torch.no_grad()
def qubit_means(model: HybridModel, records: list, device) -> np.ndarray:
    """Return mean ⟨Z_i⟩ per qubit across all records. Shape: (n_qubits,)."""
    ds     = SpectrogramDataset(records, transform=make_transform(augment=False))
    loader = make_loader(ds, cfg.BATCH_SIZE, shuffle=False)
    model.eval()
    q_all = []
    for imgs, _ in loader:
        out = model(imgs.to(device))
        if out[2] is None:
            return np.zeros(cfg.N_QUBITS)
        q_all.append(out[2].cpu().numpy())
    return np.concatenate(q_all, axis=0).mean(axis=0)


# ── 2. Parameter-shift feature importance ────────────────────────────────────

def angle_grad_importance(model: HybridModel, imgs: torch.Tensor, device) -> np.ndarray:
    """
    Estimate ∂⟨Z_total⟩/∂angle_j for each angle dimension j using finite differences
    on the VQC INPUT angles (not weights).  This is equivalent to parameter-shift
    for angle-embedded inputs.

    Returns shape (n_qubits,) — importance of each input angle / qubit dimension.
    """
    model.eval()
    if not isinstance(model, HybridModel):
        return np.zeros(cfg.N_QUBITS)

    imgs = imgs.to(device)
    shift = np.pi / 2.0

    with torch.no_grad():
        c_feats, _ = model.backbone(imgs)
        angles     = torch.tanh(model.vqc_block.proj(c_feats)) * np.pi  # (B, n_qubits)

    importances = np.zeros(cfg.N_QUBITS)
    for j in range(cfg.N_QUBITS):
        a_plus  = angles.clone(); a_plus[:, j]  += shift
        a_minus = angles.clone(); a_minus[:, j] -= shift
        with torch.no_grad():
            z_plus  = model.vqc_block.vqc(a_plus).cpu().numpy()   # (B, n_qubits)
            z_minus = model.vqc_block.vqc(a_minus).cpu().numpy()
        # Sum gradient signal across qubits and samples
        importances[j] = np.abs((z_plus - z_minus) / 2.0).mean()

    return importances


# ── 3. Gate activity report ───────────────────────────────────────────────────

@torch.no_grad()
def gate_report(model: HybridModel, records: list, device) -> dict:
    """
    For GatedFusion models, report mean gate values for genuine vs. replay.
    Returns {"genuine": mean_gate, "replay": mean_gate} (each shape n_classes).
    """
    from models.fusion import GatedFusion
    if not isinstance(getattr(model, "fusion", None), GatedFusion):
        return {}

    gates_by_class = {0: [], 1: []}
    ds     = SpectrogramDataset(records, transform=make_transform(augment=False))
    loader = make_loader(ds, cfg.BATCH_SIZE, shuffle=False)
    model.eval()
    for imgs, labels in loader:
        out        = model(imgs.to(device))
        c_logits   = out[3]
        q_logits   = out[4]
        gate = model.fusion.gate_net(
            torch.cat([c_logits, q_logits], dim=1)
        ).cpu().numpy()
        for i, lbl in enumerate(labels.numpy()):
            gates_by_class[int(lbl)].append(gate[i])

    return {
        "genuine": np.array(gates_by_class[0]).mean(axis=0).tolist() if gates_by_class[0] else None,
        "replay":  np.array(gates_by_class[1]).mean(axis=0).tolist() if gates_by_class[1] else None,
    }


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_qubit_comparison(genuine_means, replay_means, n_qubits, title, out_path):
    x   = np.arange(n_qubits)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, genuine_means, width=0.35, label="Genuine", color="steelblue")
    ax.bar(x + 0.2, replay_means,  width=0.35, label="Replay",  color="darkorange")
    ax.set_xlabel("Qubit index")
    ax.set_ylabel("Mean ⟨Z⟩")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [quantum_attr] Per-qubit ⟨Z⟩ -> {out_path}")


def plot_importance(importances, out_path):
    n   = len(importances)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(n), importances, color="seagreen")
    ax.set_xlabel("Angle dimension (qubit index)")
    ax.set_ylabel("Mean |∂⟨Z⟩/∂angle|  (parameter-shift)")
    ax.set_title("Quantum Feature Importance")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [quantum_attr] Feature importance -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_quantum_attr(ckpt_path: str, target_sat: str,
                     out_dir: str = None, device: str = None,
                     n_samples: int = 64):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = load_model_from_ckpt(ckpt, device)
    model.eval()

    if not isinstance(model, HybridModel):
        print("  [quantum_attr] Model has no quantum branch — skipping.")
        return

    sats   = ckpt.get("all_sats", cfg.REDUCED_SATELLITES)
    all_recs = discover_images(cfg.SPECTROGRAM_ROOT, sats)
    g_recs = [r for r in all_recs
               if r["satellite"] == target_sat and r["label_name"] == "genuine"][:n_samples]
    r_recs = [r for r in all_recs
               if r["satellite"] == target_sat and r["label_name"] == "replay"][:n_samples]

    out_dir = Path(out_dir or (Path(cfg.RESULTS_ROOT) / "explain"))
    out_dir.mkdir(parents=True, exist_ok=True)
    mkey    = f"{ckpt['backbone']}__{ckpt.get('fusion', 'classical')}"

    # 1. Per-qubit ⟨Z⟩
    g_means = qubit_means(model, g_recs, device)
    r_means = qubit_means(model, r_recs, device)
    plot_qubit_comparison(
        g_means, r_means, cfg.N_QUBITS,
        title=f"Per-qubit ⟨Z⟩: {target_sat} [{mkey}]",
        out_path=str(out_dir / f"qubit_z_{target_sat}_{mkey}.png"),
    )
    print(f"  Genuine qubit means: {np.round(g_means, 3)}")
    print(f"  Replay  qubit means: {np.round(r_means, 3)}")

    # 2. Parameter-shift feature importance (on a small batch)
    transform = make_transform(augment=False)
    ds        = SpectrogramDataset((g_recs + r_recs)[:32], transform=transform)
    imgs_sample = torch.stack([ds[i][0] for i in range(min(8, len(ds)))])
    importances = angle_grad_importance(model, imgs_sample, device)
    plot_importance(
        importances,
        out_path=str(out_dir / f"feature_importance_{target_sat}_{mkey}.png"),
    )

    # 3. Gate report (GatedFusion only)
    gate_info = gate_report(model, (g_recs + r_recs), device)
    if gate_info:
        print(f"  Gate (genuine): {gate_info['genuine']}")
        print(f"  Gate (replay):  {gate_info['replay']}")
        import json
        with open(out_dir / f"gate_report_{target_sat}_{mkey}.json", "w") as f:
            json.dump({"target_sat": target_sat, "model": mkey, **gate_info}, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       required=True)
    ap.add_argument("--target_sat", required=True)
    ap.add_argument("--out_dir",    default=None)
    ap.add_argument("--device",     default=None)
    ap.add_argument("--n_samples",  type=int, default=64)
    a = ap.parse_args()
    run_quantum_attr(a.ckpt, a.target_sat, a.out_dir, a.device, a.n_samples)
