"""
Merge per-sat PGD sweep CSVs from multiple parallel runs (e.g. one per GPU,
each covering a different backbone) into a single combined results/adaptive_pgd/.

Usage
-----
    python3.10 eval/merge_pgd_results.py \\
        --inputs results/adaptive_pgd_se results/adaptive_pgd_dn \\
        --out_root results
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Directories each containing adaptive_pgd/per_sat_sweep.csv")
    ap.add_argument("--out_root", default=cfg.RESULTS_ROOT)
    a = ap.parse_args()

    frames, all_rows = [], []
    for inp in a.inputs:
        csv_path = Path(inp) / "adaptive_pgd" / "per_sat_sweep.csv"
        json_path = Path(inp) / "adaptive_pgd" / "all_rows.json"
        if not csv_path.exists():
            print(f"[skip] missing {csv_path}")
            continue
        frames.append(pd.read_csv(csv_path))
        if json_path.exists():
            all_rows.extend(json.loads(json_path.read_text()))

    if not frames:
        print("No PGD results found to merge.")
        return

    df = pd.concat(frames, ignore_index=True)
    macro = df.groupby(["backbone", "fusion", "eps"]).agg(
        trr_mean=("trr", "mean"), trr_std=("trr", "std"),
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
    ).reset_index()

    pgd_dir = Path(a.out_root) / "adaptive_pgd"
    pgd_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(pgd_dir / "per_sat_sweep.csv", index=False)
    macro.to_csv(pgd_dir / "macro_curve.csv", index=False)
    with open(pgd_dir / "all_rows.json", "w") as f:
        json.dump(all_rows, f, indent=2)

    print(f"Merged {len(df)} rows from {len(frames)} sources -> {pgd_dir}")
    pd.set_option("display.float_format", "{:.3f}".format)
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
