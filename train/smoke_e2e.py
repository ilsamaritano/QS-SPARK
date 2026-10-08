"""
Smoke E2E test: 2 satellites × classical vs hybrid-sum × 2 epochs.

Verifies the full pipeline end-to-end:
  preprocessing images → dataset → model forward → training → eval → figures

Usage
-----
    python3.10 train/smoke_e2e.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from train.train_ovr import train_ovr
from eval.replay_eval import evaluate_replay_detection
from eval.ovr_eval import evaluate_ovr
from models.hybrid import model_key

SMOKE_SATS    = ["Gaofen12", "Gaofen3"]
SMOKE_EPOCHS  = cfg.EPOCHS_SMOKE
SMOKE_CONFIGS = [
    ("se_resnet34", None),          # classical baseline
    ("se_resnet34", "sum"),          # hybrid-sum
]


def main():
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(cfg.RESULTS_ROOT) / "smoke"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\nSmoke E2E  |  device={device}  |  epochs={SMOKE_EPOCHS}")
    print(f"Satellites: {SMOKE_SATS}")
    print(f"Configs:    {SMOKE_CONFIGS}\n")

    all_results = []
    for backbone, fusion in SMOKE_CONFIGS:
        mkey = model_key(backbone, fusion)
        print(f"\n{'='*60}")
        print(f"Config: {mkey}")
        print(f"{'='*60}")

        for sat in SMOKE_SATS:
            tr = train_ovr(
                target_sat=sat,
                backbone_name=backbone,
                fusion_name=fusion,
                all_satellites=SMOKE_SATS,
                seed=cfg.SEED,
                epochs=SMOKE_EPOCHS,
                device=device,
                out_root=str(out_root),
            )

            ckpt_path = Path(tr["ckpt_path"])
            rep = evaluate_replay_detection(
                target_sat=sat,
                ckpt_path=str(ckpt_path),
                all_satellites=SMOKE_SATS,
                device=device,
            )
            ovr = evaluate_ovr(
                target_sat=sat,
                ckpt_path=str(ckpt_path),
                all_satellites=SMOKE_SATS,
                device=device,
            )
            row = {
                "backbone": backbone, "fusion": fusion or "classical", "satellite": sat,
                "train_auroc": tr.get("test_auroc"),
                "replay_auroc": rep.get("auroc"),
                "replay_trr":  rep.get("trr_at_far1pct"),
                "replay_eer":  rep.get("eer"),
                "ovr_auroc":   ovr.get("auroc"),
                "ovr_tar":     ovr.get("tar_at_far1pct"),
            }
            all_results.append(row)
            print(f"  {sat}: replay_auroc={rep.get('auroc','N/A'):.3f}  "
                  f"trr@1%={rep.get('trr_at_far1pct','N/A'):.3f}  "
                  f"ovr_auroc={ovr.get('auroc','N/A'):.3f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SMOKE E2E SUMMARY")
    print("="*70)
    print(f"{'Config':<30} {'Satellite':<14} {'Replay AUROC':>12} {'TRR@1%':>8} {'OvR AUROC':>10}")
    print("-"*70)
    for r in all_results:
        cfg_label = f"{r['backbone']}__{r['fusion']}"
        rauc = f"{r['replay_auroc']:.3f}" if r["replay_auroc"] is not None else "N/A"
        trr  = f"{r['replay_trr']:.3f}"  if r["replay_trr"] is not None else "N/A"
        oauc = f"{r['ovr_auroc']:.3f}"   if r["ovr_auroc"] is not None else "N/A"
        print(f"{cfg_label:<30} {r['satellite']:<14} {rauc:>12} {trr:>8} {oauc:>10}")

    # Save
    with open(out_root / "smoke_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_root}/smoke_results.json")

    # ── Simple ROC plot ───────────────────────────────────────────────────────
    _plot_summary(all_results, out_root)


def _plot_summary(rows, out_root):
    configs = list({(r["backbone"], r["fusion"]) for r in rows})
    sats    = list({r["satellite"] for r in rows})

    fig, axes = plt.subplots(1, len(sats), figsize=(6 * len(sats), 5))
    if len(sats) == 1:
        axes = [axes]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, sat in zip(axes, sats):
        for ci, (bb, fus) in enumerate(configs):
            row = next((r for r in rows
                        if r["satellite"] == sat and r["backbone"] == bb and r["fusion"] == fus),
                       None)
            if row is None:
                continue
            label = f"{bb}__{fus}  (AUROC={row['replay_auroc']:.3f})"
            ax.bar(ci, row["replay_auroc"] or 0, color=colors[ci % len(colors)],
                   label=label, width=0.6, alpha=0.8)
        ax.set_ylim(0, 1)
        ax.set_title(f"Replay AUROC — {sat}")
        ax.set_xticks([])
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("AUROC")

    plt.tight_layout()
    fig_path = out_root / "smoke_replay_auroc.png"
    plt.savefig(str(fig_path), dpi=150)
    plt.close()
    print(f"[plot] {fig_path}")


if __name__ == "__main__":
    main()
