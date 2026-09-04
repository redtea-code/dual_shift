#!/usr/bin/env bash
set -u
ROOT=/zjs/AD_Project/dual_shift_ds043_run
JOB=$ROOT/run_ds043_job.sh
OUT=$ROOT/outputs/ds043_capm_grl
mkdir -p "$OUT/logs" "$OUT/diagnostics"
jobs=()
for variant in F0 F1 F2 F3 R1 R2 R3; do
  for seed in 42 43 44; do jobs+=("$variant $seed"); done
done
for ((i=0; i<${#jobs[@]}; i+=3)); do
  pids=()
  for gpu in 0 1 2; do
    j=$((i+gpu)); [ "$j" -ge "${#jobs[@]}" ] && continue
    read -r variant seed <<< "${jobs[$j]}"
    [ -s "$OUT/$variant/seed_$seed/report.json" ] && continue
    "$JOB" "$variant" "$seed" "$gpu" & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
done
