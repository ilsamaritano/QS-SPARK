"""
Benchmark matrix runner for PLASyR-Q.

Iterates over all (backbone, fusion_name, satellite) combinations, trains each
OvR classifier, runs Mahalanobis replay detection and OvR authentication
evaluation, and aggregates results into a CSV / JSON table.

Usage
-----
# Smoke run (2 epochs, 2 sats, classical + hybrid-sum)
python3.10 train/benchmark.py --epochs 2 --satellites Gaofen12 Gaofen3 --fusions classical sum

# Full reduced benchmark (40 epochs, all 6 sats, all fusion ops, 3 seeds)
python3.10 train/benchmark.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from train.train_ovr import train_ovr
from eval.replay_eval import evaluate_replay_detection
from eval.ovr_eval import evaluate_ovr


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--satellites", nargs="+", default=cfg.REDUCED_SATELLITES)
    ap.add_argument("--backbones",  nargs="+", default=cfg.BACKBONES)
    ap.add_argument("--fusions",    nargs="+",
                    default=["classical"] + cfg.FUSION_OPS,
                    help="'classical' means no VQC; others are fusion operator names")
    ap.add_argument("--seeds",      nargs="+", type=int, default=[cfg.SEED])
    ap.add_argument("--epochs",     type=int,  default=cfg.EPOCHS_REDUCED)
    ap.add_argument("--out_root",   default=cfg.RESULTS_ROOT)
    ap.add_argument("--device",     default=None)
    ap.add_argument("--skip_train", action="store_true",
                    help="Skip training; only aggregate existing results")
    ap.add_argument("--n_qubits",   type=int, default=cfg.N_QUBITS)
    ap.add_argument("--n_layers",   type=int, default=cfg.VQC_LAYERS)
    ap.add_argument("--quantum_lr", type=float, default=None,
                    help="Separate (higher) LR for the VQC branch; None = single shared LR (original behavior)")
    ap.add_argument("--grad_clip_norm", type=float, default=None)
    return ap.parse_args()


def run_matrix(args):
    rows = []
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    total = (len(args.backbones) * len(args.fusions) *
             len(args.satellites) * len(args.seeds))
    done  = 0

    for backbone in args.backbones:
        for fusion_arg in args.fusions:
            fusion_name = None if fusion_arg == "classical" else fusion_arg

            for seed in args.seeds:
                # ── Training phase ────────────────────────────────────────────
                if not args.skip_train:
                    print(f"\n{'='*70}")
                    print(f"  backbone={backbone}  fusion={fusion_arg}  seed={seed}")
                    print(f"{'='*70}")
                    for sat in args.satellites:
                        done += 1
                        print(f"\n[{done}/{total}] Training OvR: {sat}")
                        train_ovr(
                            target_sat=sat,
                            backbone_name=backbone,
                            fusion_name=fusion_name,
                            all_satellites=args.satellites,
                            seed=seed,
                            epochs=args.epochs,
                            device=args.device,
                            out_root=str(out_root),
                            n_qubits=args.n_qubits,
                            n_layers=args.n_layers,
                            quantum_lr=args.quantum_lr,
                            grad_clip_norm=args.grad_clip_norm,
                        )

                # ── Evaluation phase ──────────────────────────────────────────
                for sat in args.satellites:
                    from models.hybrid import model_key
                    mkey      = model_key(backbone, fusion_name)
                    ckpt_path = out_root / "ovr" / sat / mkey / f"seed{seed}" / "best.pt"

                    if not ckpt_path.exists():
                        print(f"  [skip] No checkpoint: {ckpt_path}")
                        continue

                    # Replay detection
                    replay_res = evaluate_replay_detection(
                        target_sat=sat,
                        ckpt_path=str(ckpt_path),
                        all_satellites=args.satellites,
                        device=args.device,
                    )

                    # OvR authentication
                    ovr_res = evaluate_ovr(
                        target_sat=sat,
                        ckpt_path=str(ckpt_path),
                        all_satellites=args.satellites,
                        device=args.device,
                    )

                    row = {
                        "backbone":          backbone,
                        "fusion":            fusion_arg,
                        "satellite":         sat,
                        "seed":              seed,
                        # Replay detection
                        "replay_auroc":      replay_res.get("auroc"),
                        "replay_eer":        replay_res.get("eer"),
                        "replay_trr_far1":   replay_res.get("trr_at_far1pct"),
                        # OvR authentication
                        "ovr_auroc":         ovr_res.get("auroc"),
                        "ovr_eer":           ovr_res.get("eer"),
                        "ovr_tar_far1":      ovr_res.get("tar_at_far1pct"),
                    }
                    rows.append(row)
                    print(f"  {sat} [{mkey}] seed={seed} | "
                          f"replay_auroc={replay_res.get('auroc', 'N/A'):.3f} "
                          f"trr@1%={replay_res.get('trr_at_far1pct', 'N/A'):.3f} | "
                          f"ovr_auroc={ovr_res.get('auroc', 'N/A'):.3f}")

    if not rows:
        print("No results collected.")
        return

    df = pd.DataFrame(rows)

    # ── Macro aggregation (mean ± std across satellites and seeds) ────────────
    agg = df.groupby(["backbone", "fusion"]).agg(
        replay_auroc_mean=("replay_auroc", "mean"),
        replay_auroc_std=("replay_auroc", "std"),
        replay_trr_mean=("replay_trr_far1", "mean"),
        replay_trr_std=("replay_trr_far1", "std"),
        replay_eer_mean=("replay_eer", "mean"),
        ovr_auroc_mean=("ovr_auroc", "mean"),
        ovr_auroc_std=("ovr_auroc", "std"),
        ovr_tar_mean=("ovr_tar_far1", "mean"),
        ovr_tar_std=("ovr_tar_far1", "std"),
    ).reset_index()

    # ── Save results ──────────────────────────────────────────────────────────
    df.to_csv(out_root / "benchmark_per_sat.csv", index=False)
    agg.to_csv(out_root / "benchmark_macro.csv", index=False)
    with open(out_root / "benchmark_per_sat.json", "w") as f:
        json.dump(rows, f, indent=2)

    print("\n" + "="*70)
    print("MACRO BENCHMARK TABLE")
    print("="*70)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(agg.to_string(index=False))
    print(f"\nResults saved to {out_root}")
    return df, agg


if __name__ == "__main__":
    args = parse_args()
    run_matrix(args)
