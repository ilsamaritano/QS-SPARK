#!/bin/bash
# Rerun of the backbone-benchmark half that crashed: GPU1 was still occupied by
# another user's job at launch, so RCM1/RCM3/Sentinel1A never trained. Runs the
# missing 45 configs on GPU0 only, then the authoritative single-process merge.
set -uo pipefail

ROOT=/home/vsammartino/plasyr-quantum
PY=/home/vsammartino/progetto_analisi/venv/bin/python3.10
LOGDIR=/tmp/claude-1088/-home-vsammartino/afb3a317-3f3a-4d4e-9fdc-6a2f2607e73a/scratchpad
RESULTS=$ROOT/results_backbones

BACKBONES="resnet18 resnet34 efficientnet_b0 mobilenetv3 vit_small"
FUSIONS="classical concat_linear product"
MISSING_SATS="RCM1 RCM3 Sentinel1A"
ALL_SATS="Gaofen12 Gaofen12-02 Gaofen3 RCM1 RCM3 Sentinel1A"

cd "$ROOT"
STATUS="$LOGDIR/backbone_bench_status.log"
echo "[$(date -Is)] Rerunning missing half (RCM1 RCM3 Sentinel1A) on GPU0" >> "$STATUS"

CUDA_VISIBLE_DEVICES=0 $PY -u train/benchmark.py \
  --epochs 40 --satellites $MISSING_SATS --backbones $BACKBONES \
  --fusions $FUSIONS --seeds 42 --out_root "$RESULTS" \
  > "$LOGDIR/bb_gpu1half.log" 2>&1
echo "[$(date -Is)] Missing-half training done (rc=$?)" | tee -a "$STATUS"

CUDA_VISIBLE_DEVICES=0 $PY -u train/benchmark.py \
  --epochs 40 --satellites $ALL_SATS --backbones $BACKBONES \
  --fusions $FUSIONS --seeds 42 --out_root "$RESULTS" --skip_train \
  > "$LOGDIR/bb_merge.log" 2>&1
echo "[$(date -Is)] Full merge done (rc=$?)" | tee -a "$STATUS"

{
  echo "===================================================================="
  echo "BACKBONE BENCHMARK (full 6 sats) COMPLETE: $(date -Is)"
  echo "===================================================================="
  cat "$RESULTS/benchmark_macro.csv"
} | tee -a "$STATUS"

touch "$LOGDIR/BACKBONE_BENCH_FULL_DONE"
