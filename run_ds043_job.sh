#!/usr/bin/env bash
set -u
ROOT=/zjs/AD_Project/dual_shift_ds043_run
PY=/opt/conda/envs/cyh/bin/python
CFG=$ROOT/journal_ds043_capm_concat_scan_filtered_1p5t_mci_ad.yaml
OUT=$ROOT/outputs/ds043_capm_grl
variant=$1; seed=$2; gpu=$3
job="${variant}_seed_${seed}_gpu_${gpu}"
dir="$OUT/$variant/seed_$seed"; log="$OUT/logs/${job}.log"; diag="$OUT/diagnostics"
hb="$diag/${job}.heartbeat.jsonl"; rec="$diag/${job}.exit.json"
mkdir -p "$dir" "$diag" "$OUT/logs"
: > "$log"; : > "$hb"
start=$(date +%s); start_iso=$(date -Is)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$gpu"
"$PY" -u -X faulthandler "$ROOT/experiments/run_capm_frequency_grl.py" --config "$CFG" --variant "$variant" --direction ADNI_to_NACC --seed "$seed" --epochs 50 --batch-size 4 --output "$dir/report.json" >> "$log" 2>&1 &
child=$!
printf '{"event":"started","variant":"%s","seed":%s,"gpu":%s,"pid":%s,"start":"%s"}\n' "$variant" "$seed" "$gpu" "$child" "$start_iso" > "$rec"
while kill -0 "$child" 2>/dev/null; do
  now=$(date -Is)
  gpu_status=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader -i "$gpu" 2>/dev/null | tr '\n' ' ')
  io_status=$(cat "/proc/$child/io" 2>/dev/null | tr '\n' ' ' || true)
  proc_state=$(awk '{print $3}' "/proc/$child/stat" 2>/dev/null || echo gone)
  cpu=$(ps -p "$child" -o %cpu= 2>/dev/null | tr -d ' ' || true)
  printf '{"time":"%s","pid":%s,"gpu":%s,"cpu_percent":"%s","state":"%s","gpu_status":"%s","io":"%s"}\n' "$now" "$child" "$gpu" "$cpu" "$proc_state" "$gpu_status" "$io_status" >> "$hb"
  sleep 30
done
rc=0; wait "$child" || rc=$?
end=$(date -Is)
if [ "$rc" -eq 0 ]; then state=completed; elif [ "$rc" -ge 128 ]; then state="signal_$((rc-128))"; else state="exit_$rc"; fi
printf '{"event":"finished","variant":"%s","seed":%s,"gpu":%s,"pid":%s,"exit_code":%s,"state":"%s","end":"%s","elapsed_seconds":%s,"report_exists":%s}\n' "$variant" "$seed" "$gpu" "$child" "$rc" "$state" "$end" "$(( $(date +%s)-start ))" "$([ -s "$dir/report.json" ] && echo true || echo false)" >> "$rec"
exit "$rc"
