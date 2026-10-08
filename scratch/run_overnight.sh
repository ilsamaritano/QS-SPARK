#!/bin/bash
# PLASyR-Q overnight full pipeline: training (2-GPU split) -> eval merge ->
# adaptive PGD sweep (2-GPU split) -> merge -> explainability -> paper figures.
#
# All phases are chained; if any phase's processes exit non-zero, the script
# still proceeds (best-effort) but logs the failure, so a single satellite
# crash doesn't block the rest of the overnight run. The DONE sentinel at the
# end always tells you what actually completed.
set -uo pipefail

ROOT=/home/vsammartino/plasyr-quantum
PY=/home/vsammartino/progetto_analisi/venv/bin/python3.10
LOGDIR=/tmp/claude-1088/-home-vsammartino/afb3a317-3f3a-4d4e-9fdc-6a2f2607e73a/scratchpad
RESULTS=$ROOT/results

SATS="Gaofen12 Gaofen12-02 Gaofen3 RCM1 RCM3 Sentinel1A"
FUSIONS="classical concat_linear sum weighted product gated max"
PGD_FUSIONS="classical sum gated product"
EXPLAIN_FUSIONS="classical sum gated product"
EXPLAIN_SATS="Gaofen12 Gaofen3"

cd "$ROOT"
STATUS="$LOGDIR/overnight_status.log"
echo "[$(date -Is)] Overnight run started" > "$STATUS"

# ── Phase 1: parallel training, split by backbone across both GPUs ─────────
echo "[$(date -Is)] Phase 1: training (GPU0=se_resnet34, GPU1=densenet121)" | tee -a "$STATUS"

CUDA_VISIBLE_DEVICES=0 $PY -u train/benchmark.py \
  --epochs 40 --satellites $SATS --backbones se_resnet34 \
  --fusions $FUSIONS --seeds 42 --out_root "$RESULTS" \
  > "$LOGDIR/phase1_gpu0_se.log" 2>&1 &
P1_GPU0=$!

CUDA_VISIBLE_DEVICES=1 $PY -u train/benchmark.py \
  --epochs 40 --satellites $SATS --backbones densenet121 \
  --fusions $FUSIONS --seeds 42 --out_root "$RESULTS" \
  > "$LOGDIR/phase1_gpu1_dn.log" 2>&1 &
P1_GPU1=$!

wait $P1_GPU0; RC0=$?
wait $P1_GPU1; RC1=$?
echo "[$(date -Is)] Phase 1 done (gpu0_rc=$RC0 gpu1_rc=$RC1)" | tee -a "$STATUS"

# ── Phase 1.5: single-process re-evaluation -> authoritative benchmark CSV ──
# (Phase 1's two parallel processes may have raced on benchmark_per_sat.csv;
#  this re-evaluates the full matrix from checkpoints, single writer, no race.)
echo "[$(date -Is)] Phase 1.5: merge/re-evaluate full benchmark matrix" | tee -a "$STATUS"
$PY -u train/benchmark.py \
  --epochs 40 --satellites $SATS --backbones se_resnet34 densenet121 \
  --fusions $FUSIONS --seeds 42 --out_root "$RESULTS" --skip_train \
  > "$LOGDIR/phase1_5_merge_eval.log" 2>&1
echo "[$(date -Is)] Phase 1.5 done (rc=$?)" | tee -a "$STATUS"

# ── Phase 2: parallel adaptive PGD sweep, split by backbone across GPUs ────
echo "[$(date -Is)] Phase 2: adaptive PGD sweep (GPU0=se_resnet34, GPU1=densenet121)" | tee -a "$STATUS"

CUDA_VISIBLE_DEVICES=0 $PY -u eval/adaptive_pgd.py \
  --backbones se_resnet34 --fusions $PGD_FUSIONS --satellites $SATS \
  --seeds 42 --epsilons 0 0.02 0.05 0.10 0.20 0.30 \
  --pgd_steps 20 --max_attack 100 \
  --out_root "$RESULTS/adaptive_pgd_se" --ckpt_root "$RESULTS" \
  > "$LOGDIR/phase2_gpu0_pgd_se.log" 2>&1 &
P2_GPU0=$!

CUDA_VISIBLE_DEVICES=1 $PY -u eval/adaptive_pgd.py \
  --backbones densenet121 --fusions $PGD_FUSIONS --satellites $SATS \
  --seeds 42 --epsilons 0 0.02 0.05 0.10 0.20 0.30 \
  --pgd_steps 20 --max_attack 100 \
  --out_root "$RESULTS/adaptive_pgd_dn" --ckpt_root "$RESULTS" \
  > "$LOGDIR/phase2_gpu1_pgd_dn.log" 2>&1 &
P2_GPU1=$!

wait $P2_GPU0; RC2=$?
wait $P2_GPU1; RC3=$?
echo "[$(date -Is)] Phase 2 done (gpu0_rc=$RC2 gpu1_rc=$RC3)" | tee -a "$STATUS"

echo "[$(date -Is)] Phase 2.5: merge PGD results" | tee -a "$STATUS"
$PY -u eval/merge_pgd_results.py \
  --inputs "$RESULTS/adaptive_pgd_se" "$RESULTS/adaptive_pgd_dn" \
  --out_root "$RESULTS" \
  > "$LOGDIR/phase2_5_merge_pgd.log" 2>&1
echo "[$(date -Is)] Phase 2.5 done (rc=$?)" | tee -a "$STATUS"

# ── Phase 3: explainability (single GPU, fast) ───────────────────────────────
echo "[$(date -Is)] Phase 3: explainability matrix (se_resnet34, $EXPLAIN_SATS)" | tee -a "$STATUS"
CUDA_VISIBLE_DEVICES=0 $PY -u explain/run_explain_matrix.py \
  --backbone se_resnet34 --fusions $EXPLAIN_FUSIONS --satellites $EXPLAIN_SATS \
  --seed 42 --ckpt_root "$RESULTS" \
  > "$LOGDIR/phase3_explain.log" 2>&1
echo "[$(date -Is)] Phase 3 done (rc=$?)" | tee -a "$STATUS"

# ── Phase 4: paper figures ───────────────────────────────────────────────────
echo "[$(date -Is)] Phase 4: generate paper figures" | tee -a "$STATUS"
$PY -u paper/generate_figures.py --results_root "$RESULTS" \
  --fig_dir "$ROOT/paper/figures" --tsne_sat Gaofen12 \
  > "$LOGDIR/phase4_figures.log" 2>&1
echo "[$(date -Is)] Phase 4 done (rc=$?)" | tee -a "$STATUS"

# ── Summary ───────────────────────────────────────────────────────────────────
{
  echo ""
  echo "===================================================================="
  echo "OVERNIGHT RUN COMPLETE: $(date -Is)"
  echo "===================================================================="
  echo "Checkpoints trained: $(find "$RESULTS/ovr" -name best.pt 2>/dev/null | wc -l) / 84"
  echo "Benchmark CSV: $RESULTS/benchmark_per_sat.csv"
  [ -f "$RESULTS/benchmark_macro.csv" ] && echo "--- benchmark_macro.csv ---" && cat "$RESULTS/benchmark_macro.csv"
  echo ""
  echo "PGD sweep: $RESULTS/adaptive_pgd/macro_curve.csv"
  [ -f "$RESULTS/adaptive_pgd/macro_curve.csv" ] && cat "$RESULTS/adaptive_pgd/macro_curve.csv"
  echo ""
  echo "Figures: $(find "$ROOT/paper/figures" -type f 2>/dev/null | wc -l) files in $ROOT/paper/figures"
  echo "Explain panels: $(find "$RESULTS/explain" -type f 2>/dev/null | wc -l) files in $RESULTS/explain"
} | tee -a "$STATUS"

touch "$LOGDIR/OVERNIGHT_DONE"
echo "[$(date -Is)] Sentinel written: $LOGDIR/OVERNIGHT_DONE" | tee -a "$STATUS"
