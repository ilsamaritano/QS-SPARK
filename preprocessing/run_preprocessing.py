"""
Walk replay_attacks/, label each pass genuine/replay by folder, and write
spectrogram PNGs to data_processed/<class>/<satellite>/<pass>/.

class = "genuine" (from original/original/) or "replay" (per-satellite
folders). Usage:

    python3.10 run_preprocessing.py --max_images_per_file 8   # checkpoint
    python3.10 run_preprocessing.py --max_images_per_file 120 # reduced run
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from preprocessing.pipeline import process_file_to_images, save_images


def discover_passes(satellites, include_replay_only=False):
    """Returns list of (satellite, label, raw_path, pass_name)."""
    items = []
    sats = list(satellites)
    if include_replay_only:
        sats += cfg.REPLAY_ONLY_SATELLITES

    for sat in sats:
        replay_dir = os.path.join(cfg.RAW_DATA_ROOT, sat)
        if os.path.isdir(replay_dir):
            for fname in sorted(os.listdir(replay_dir)):
                fpath = os.path.join(replay_dir, fname)
                if os.path.isfile(fpath):
                    items.append((sat, "replay", fpath, fname))

        genuine_dir = os.path.join(cfg.RAW_DATA_ROOT, cfg.GENUINE_SUBDIR)
        if os.path.isdir(genuine_dir):
            for fname in sorted(os.listdir(genuine_dir)):
                if fname.startswith(sat + "_"):
                    fpath = os.path.join(genuine_dir, fname)
                    if os.path.isfile(fpath):
                        items.append((sat, "genuine", fpath, fname))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--satellites", nargs="+", default=cfg.REDUCED_SATELLITES)
    ap.add_argument("--include_replay_only", action="store_true")
    ap.add_argument("--max_images_per_file", type=int, default=cfg.MAX_IMAGES_PER_FILE)
    ap.add_argument("--out_root", default=cfg.SPECTROGRAM_ROOT)
    args = ap.parse_args()

    items = discover_passes(args.satellites, args.include_replay_only)
    print(f"Discovered {len(items)} raw passes across {len(args.satellites)} satellites")

    summary = []
    for sat, label, fpath, fname in items:
        t0 = time.time()
        pass_id = os.path.splitext(fname)[0]
        out_dir = os.path.join(args.out_root, label, sat, pass_id)
        try:
            images = process_file_to_images(fpath, max_images=args.max_images_per_file)
        except Exception as e:
            print(f"[ERROR] {fpath}: {e}")
            continue
        n = save_images(images, out_dir, prefix=pass_id)
        dt = time.time() - t0
        summary.append((sat, label, pass_id, n, dt))
        print(f"[{label:7s}] {sat:12s} {pass_id:30s} -> {n:4d} images  ({dt:5.1f}s)")

    print("\n=== Summary ===")
    by_class = {}
    for sat, label, pass_id, n, dt in summary:
        by_class.setdefault((sat, label), 0)
        by_class[(sat, label)] += n
    for (sat, label), n in sorted(by_class.items()):
        print(f"  {sat:12s} {label:7s} total_images={n}")


if __name__ == "__main__":
    main()
