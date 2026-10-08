#!/bin/bash
# Both GPUs are currently held by another user's job (fmitola, ~48GB each).
# Poll every 10 min; when BOTH GPUs have >= 24 GB free, launch the backbone
# benchmark. Gives up after 72 h.
set -u
LOGDIR=/tmp/claude-1088/-home-vsammartino/afb3a317-3f3a-4d4e-9fdc-6a2f2607e73a/scratchpad
STATUS="$LOGDIR/backbone_bench_status.log"
echo "[$(date -Is)] Waiting for GPUs to free up (another user's job is running)" > "$STATUS"

for i in $(seq 1 432); do   # 432 * 10 min = 72 h
  free0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  free1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
  if [ "$free0" -ge 24000 ] && [ "$free1" -ge 24000 ]; then
    echo "[$(date -Is)] GPUs free (gpu0=${free0}MiB gpu1=${free1}MiB) — launching benchmark" >> "$STATUS"
    bash /home/vsammartino/plasyr-quantum/scratch/run_backbone_benchmark.sh
    exit $?
  fi
  sleep 600
done
echo "[$(date -Is)] Gave up after 72h — GPUs never freed" >> "$STATUS"
exit 1
