#!/bin/bash
# Controlled ablation: isolate the quantum-specific optimization effect from
# the epoch-count confound found in run_optimized.sh (60 epochs alone hurt
# classical models too, via overfitting on a tiny validation set -- train AUC
# hits 1.0 by epoch ~15-25 regardless of quantum changes).
#
# This run holds epochs=40 FIXED (matching the original baseline exactly) and
# changes ONLY: separate quantum-branch LR (5e-3), gradient clipping (1.0),
# and bigger VQC (8 qubits x 4 layers vs 6x3). Classical is skipped entirely
# since it is bit-for-bit identical to the existing baseline (quantum_lr /
# n_qubits / n_layers have no effect on ClassicalOnlyModel).
set -uo pipefail

ROOT=/home/vsammartino/plasyr-quantum
PY=/home/vsammartino/progetto_analisi/venv/bin/python3.10
LOGDIR=/tmp/claude-1088/-home-vsammartino/afb3a317-3f3a-4d4e-9fdc-6a2f2607e73a/scratchpad
RESULTS=$ROOT/results_optimized_v2

FUSIONS="concat_linear sum gated product"
SATS_GPU0="Gaofen12 Gaofen12-02 Gaofen3"
SATS_GPU1="RCM1 RCM3 Sentinel1A"
ALL_SATS="Gaofen12 Gaofen12-02 Gaofen3 RCM1 RCM3 Sentinel1A"

cd "$ROOT"
STATUS="$LOGDIR/optimized_v2_status.log"
echo "[$(date -Is)] Optimized v2 (controlled ablation) started" > "$STATUS"

echo "[$(date -Is)] Phase 1: training, epochs=40 (matches baseline), quantum-only changes" | tee -a "$STATUS"

CUDA_VISIBLE_DEVICES=0 $PY -u train/benchmark.py \
  --epochs 40 --satellites $SATS_GPU0 --backbones se_resnet34 \
  --fusions $FUSIONS --seeds 42 --out_root "$RESULTS" \
  --n_qubits 8 --n_layers 4 --quantum_lr 5e-3 --grad_clip_norm 1.0 \
  > "$LOGDIR/opt_v2_gpu0.log" 2>&1 &
P0=$!

CUDA_VISIBLE_DEVICES=1 $PY -u train/benchmark.py \
  --epochs 40 --satellites $SATS_GPU1 --backbones se_resnet34 \
  --fusions $FUSIONS --seeds 42 --out_root "$RESULTS" \
  --n_qubits 8 --n_layers 4 --quantum_lr 5e-3 --grad_clip_norm 1.0 \
  > "$LOGDIR/opt_v2_gpu1.log" 2>&1 &
P1=$!

wait $P0; RC0=$?
wait $P1; RC1=$?
echo "[$(date -Is)] Phase 1 done (gpu0_rc=$RC0 gpu1_rc=$RC1)" | tee -a "$STATUS"

echo "[$(date -Is)] Phase 1.5: merge/re-evaluate full matrix" | tee -a "$STATUS"
$PY -u train/benchmark.py \
  --epochs 40 --satellites $ALL_SATS --backbones se_resnet34 \
  --fusions $FUSIONS --seeds 42 --out_root "$RESULTS" --skip_train \
  > "$LOGDIR/opt_v2_merge_eval.log" 2>&1
echo "[$(date -Is)] Phase 1.5 done (rc=$?)" | tee -a "$STATUS"

{
  echo ""
  echo "===================================================================="
  echo "OPTIMIZED V2 (controlled) RUN COMPLETE: $(date -Is)"
  echo "===================================================================="
  [ -f "$RESULTS/benchmark_macro.csv" ] && cat "$RESULTS/benchmark_macro.csv"
} | tee -a "$STATUS"

touch "$LOGDIR/OPTIMIZED_V2_DONE"
echo "[$(date -Is)] Sentinel written" | tee -a "$STATUS"
