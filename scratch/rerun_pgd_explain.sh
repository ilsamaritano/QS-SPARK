#!/bin/bash
# Resume run: Phase 1/1.5 (training + benchmark eval) already succeeded in the
# first overnight pass (84/84 checkpoints, valid benchmark_per_sat.csv).
# This reruns only the phases that crashed due to two bugs since fixed:
#   1. GaussianMahalanobis PCA projection wasn't mirrored in the differentiable
#      torch scorer used by PGD (eval/adaptive_pgd.py _maha_torch).
#   2. HybridModel.get_joint_embedding() hard-wrapped torch.no_grad(), silently
#      blocking gradient flow needed by the PGD attack (models/hybrid.py).
#   3. Grad-CAM output wasn't detached before .numpy() (explain/gradcam.py).
set -uo pipefail

ROOT=/home/vsammartino/plasyr-quantum
PY=/home/vsammartino/progetto_analisi/venv/bin/python3.10
LOGDIR=/tmp/claude-1088/-home-vsammartino/afb3a317-3f3a-4d4e-9fdc-6a2f2607e73a/scratchpad
RESULTS=$ROOT/results

SATS="Gaofen12 Gaofen12-02 Gaofen3 RCM1 RCM3 Sentinel1A"
PGD_FUSIONS="classical sum gated product"
EXPLAIN_FUSIONS="classical sum gated product"
EXPLAIN_SATS="Gaofen12 Gaofen3"

cd "$ROOT"
STATUS="$LOGDIR/rerun_status.log"
echo "[$(date -Is)] Rerun (PGD + explain) started" > "$STATUS"

rm -rf "$RESULTS/adaptive_pgd_se" "$RESULTS/adaptive_pgd_dn" "$RESULTS/adaptive_pgd" "$RESULTS/explain"

echo "[$(date -Is)] Phase 2: adaptive PGD sweep (GPU0=se_resnet34, GPU1=densenet121)" | tee -a "$STATUS"

CUDA_VISIBLE_DEVICES=0 $PY -u eval/adaptive_pgd.py \
  --backbones se_resnet34 --fusions $PGD_FUSIONS --satellites $SATS \
  --seeds 42 --epsilons 0 0.02 0.05 0.10 0.20 0.30 \
  --pgd_steps 20 --max_attack 100 \
  --out_root "$RESULTS/adaptive_pgd_se" --ckpt_root "$RESULTS" \
  > "$LOGDIR/rerun_gpu0_pgd_se.log" 2>&1 &
P2_GPU0=$!

CUDA_VISIBLE_DEVICES=1 $PY -u eval/adaptive_pgd.py \
  --backbones densenet121 --fusions $PGD_FUSIONS --satellites $SATS \
  --seeds 42 --epsilons 0 0.02 0.05 0.10 0.20 0.30 \
  --pgd_steps 20 --max_attack 100 \
  --out_root "$RESULTS/adaptive_pgd_dn" --ckpt_root "$RESULTS" \
  > "$LOGDIR/rerun_gpu1_pgd_dn.log" 2>&1 &
P2_GPU1=$!

wait $P2_GPU0; RC2=$?
wait $P2_GPU1; RC3=$?
echo "[$(date -Is)] Phase 2 done (gpu0_rc=$RC2 gpu1_rc=$RC3)" | tee -a "$STATUS"

echo "[$(date -Is)] Phase 2.5: merge PGD results" | tee -a "$STATUS"
$PY -u eval/merge_pgd_results.py \
  --inputs "$RESULTS/adaptive_pgd_se" "$RESULTS/adaptive_pgd_dn" \
  --out_root "$RESULTS" \
  > "$LOGDIR/rerun_merge_pgd.log" 2>&1
echo "[$(date -Is)] Phase 2.5 done (rc=$?)" | tee -a "$STATUS"

echo "[$(date -Is)] Phase 3: explainability matrix (se_resnet34, $EXPLAIN_SATS)" | tee -a "$STATUS"
CUDA_VISIBLE_DEVICES=0 $PY -u explain/run_explain_matrix.py \
  --backbone se_resnet34 --fusions $EXPLAIN_FUSIONS --satellites $EXPLAIN_SATS \
  --seed 42 --ckpt_root "$RESULTS" \
  > "$LOGDIR/rerun_explain.log" 2>&1
echo "[$(date -Is)] Phase 3 done (rc=$?)" | tee -a "$STATUS"

{
  echo ""
  echo "===================================================================="
  echo "RERUN COMPLETE: $(date -Is)"
  echo "===================================================================="
  echo "PGD sweep: $RESULTS/adaptive_pgd/macro_curve.csv"
  [ -f "$RESULTS/adaptive_pgd/macro_curve.csv" ] && cat "$RESULTS/adaptive_pgd/macro_curve.csv"
  echo ""
  echo "Explain panels: $(find "$RESULTS/explain" -type f 2>/dev/null | wc -l) files in $RESULTS/explain"
} | tee -a "$STATUS"

rm -f "$LOGDIR/RERUN_DONE"
touch "$LOGDIR/RERUN_DONE"
echo "[$(date -Is)] Sentinel written: $LOGDIR/RERUN_DONE" | tee -a "$STATUS"
