"""
Loop wrapper around gradcam.run_gradcam and quantum_attr.run_quantum_attr
across a representative subset of trained checkpoints.

Running explainability on the full 84-config benchmark matrix would produce
an unmanageable number of figures; instead this covers a fixed, paper-relevant
subset: one backbone, a handful of fusion operators (including the classical
baseline for contrast), on two satellites.

Usage
-----
    python3.10 explain/run_explain_matrix.py \\
        --backbone se_resnet34 --fusions classical sum gated product \\
        --satellites Gaofen12 Gaofen3 --ckpt_root results
"""
import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from models.hybrid import model_key
from explain.gradcam import run_gradcam
from explain.quantum_attr import run_quantum_attr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone",   default="se_resnet34")
    ap.add_argument("--fusions",    nargs="+", default=["classical", "sum", "gated", "product"])
    ap.add_argument("--satellites", nargs="+", default=["Gaofen12", "Gaofen3"])
    ap.add_argument("--seed",       type=int, default=cfg.SEED)
    ap.add_argument("--ckpt_root",  default=cfg.RESULTS_ROOT)
    ap.add_argument("--out_dir",    default=None)
    ap.add_argument("--device",     default=None)
    a = ap.parse_args()

    ckpt_root = Path(a.ckpt_root)
    out_dir   = a.out_dir or str(Path(cfg.RESULTS_ROOT) / "explain")

    n_ok, n_skip, n_fail = 0, 0, 0
    for fusion_arg in a.fusions:
        fusion_name = None if fusion_arg == "classical" else fusion_arg
        mkey = model_key(a.backbone, fusion_name)

        for sat in a.satellites:
            ckpt_path = ckpt_root / "ovr" / sat / mkey / f"seed{a.seed}" / "best.pt"
            if not ckpt_path.exists():
                print(f"[skip] no checkpoint: {ckpt_path}")
                n_skip += 1
                continue

            print(f"\n=== {sat} [{mkey}] ===")
            try:
                # Grad-CAM on both classes (genuine attribution + replay attribution)
                run_gradcam(str(ckpt_path), sat, label=0, n_samples=16,
                           out_dir=out_dir, device=a.device)
                run_gradcam(str(ckpt_path), sat, label=1, n_samples=16,
                           out_dir=out_dir, device=a.device)

                # Quantum attribution only meaningful for hybrid models
                if fusion_name is not None:
                    run_quantum_attr(str(ckpt_path), sat,
                                     out_dir=out_dir, device=a.device, n_samples=64)
                n_ok += 1
            except Exception as e:
                print(f"[FAIL] {sat} [{mkey}]: {e}")
                traceback.print_exc()
                n_fail += 1

    print(f"\nExplainability matrix done: ok={n_ok} skip={n_skip} fail={n_fail}")


if __name__ == "__main__":
    main()
