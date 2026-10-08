#!/bin/bash
# Backbone benchmark for the NDSS submission: motivate the SE-ResNet-34 choice.
# 5 additional backbones (resnet18, resnet34, efficientnet_b0, mobilenetv3,
# vit_small) x {classical, concat_linear, product} x 6 satellites, epochs=40,
# seed=42 — protocol identical to the main benchmark. Satellites split across
# both GPUs, then a single-process --skip_train pass regenerates the CSVs
# authoritatively (avoids the parallel-writer race).
set -uo pipefail

ROOT=/home/vsammartino/plasyr-quantum
PY=/home/vsammartino/progetto_analisi/venv/bin/python3.10
LOGDIR=/tmp/claude-1088/-home-vsammartino/afb3a317-3f3a-4d4e-9fdc-6a2f2607e73a/scratchpad
RESULTS=$ROOT/results_backbones

BACKBONES="resnet18 resnet34 efficientnet_b0 mobilenetv3 vit_small"
FUSIONS="classical concat_linear product"
SATS_GPU0="Gaofen12 Gaofen12-02 Gaofen3"
SATS_GPU1="RCM1 RCM3 Sentinel1A"
ALL_SATS="Gaofen12 Gaofen12-02 Gaofen3 RCM1 RCM3 Sentinel1A"

cd "$ROOT"
STATUS="$LOGDIR/backbone_bench_status.log"
echo "[$(date -Is)] Backbone benchmark started" > "$STATUS"

CUDA_VISIBLE_DEVICES=0 $PY -u train/benchmark.py \
  --epochs 40 --satellites $SATS_GPU0 --backbones $BACKBONES \
  --fusions $FUSIONS --seeds 42 --out_root "$RESULTS" \
  > "$LOGDIR/bb_gpu0.log" 2>&1 &
P0=$!

CUDA_VISIBLE_DEVICES=1 $PY -u train/benchmark.py \
  --epochs 40 --satellites $SATS_GPU1 --backbones $BACKBONES \
  --fusions $FUSIONS --seeds 42 --out_root "$RESULTS" \
  > "$LOGDIR/bb_gpu1.log" 2>&1 &
P1=$!

wait $P0; RC0=$?
wait $P1; RC1=$?
echo "[$(date -Is)] Training done (gpu0_rc=$RC0 gpu1_rc=$RC1)" | tee -a "$STATUS"

$PY -u train/benchmark.py \
  --epochs 40 --satellites $ALL_SATS --backbones $BACKBONES \
  --fusions $FUSIONS --seeds 42 --out_root "$RESULTS" --skip_train \
  > "$LOGDIR/bb_merge.log" 2>&1
echo "[$(date -Is)] Merge/eval done (rc=$?)" | tee -a "$STATUS"

{
  echo "===================================================================="
  echo "BACKBONE BENCHMARK COMPLETE: $(date -Is)"
  echo "===================================================================="
  [ -f "$RESULTS/benchmark_macro.csv" ] && cat "$RESULTS/benchmark_macro.csv"
} | tee -a "$STATUS"

touch "$LOGDIR/BACKBONE_BENCH_DONE"
