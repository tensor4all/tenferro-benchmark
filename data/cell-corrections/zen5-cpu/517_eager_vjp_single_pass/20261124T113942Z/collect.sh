#!/usr/bin/env bash
# tenferro-benchmark #517: eager grad_sum_{svd_s,eigh}_vjp at 4T, old (defective,
# PR #517 head debcfae) vs new (single-pass) harness, same pin ce9125d0, alternating
# process starts. Numbers here are diagnostics for the #517 cells only.
set -uo pipefail
H=~/work/tenferro-benchmark-517
OUT=~/work/517-ab/runs/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
export BENCHMARK_TARGET_PROFILE=zen5-cpu PUBLICATION_GATE_PROFILE=full \
       PUBLICATION_GATE_SUITE=large PUBLICATION_GATE_TENFERRO_MODE=eager \
       CPU_OPS_BENCHMARK_FILTER=grad_sum_svd_s_vjp,grad_sum_eigh_vjp
source "$H/scripts/thread_env.sh"
source "$H/scripts/benchmark_host_idle.sh"
configure_cpu_thread_env 4 || exit 1
echo "affinity: $(taskset -pc $$)  measured=$BENCHMARK_MEASURED_CPUS observer=$BENCHMARK_OBSERVER_CPUS" | tee "$OUT/meta.txt"
echo "boost=$(cat /sys/devices/system/cpu/cpufreq/boost) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)" | tee -a "$OUT/meta.txt"
sha256sum ~/work/517-ab/publication_gate.{old,new} >> "$OUT/meta.txt"
STARTS=${STARTS:-6}
for i in $(seq 1 "$STARTS"); do
  if (( i % 2 )); then order="old new"; else order="new old"; fi
  for arm in $order; do
    tries=0
    until assert_benchmark_host_idle 2>>"$OUT/idle.log"; do
      tries=$((tries+1)); (( tries > 60 )) && { echo "idle gate refused 60x"; exit 1; }
      sleep 30
    done
    echo "$(date -u +%T) start=$i arm=$arm idle_retries=$tries uptime=$(uptime | sed 's/.*average: //')" | tee -a "$OUT/meta.txt"
    CPU_OPS_RAW_SAMPLES="$OUT/raw_${arm}_${i}.jsonl" ~/work/517-ab/publication_gate.$arm > "$OUT/rows_${arm}_${i}.csv" 2>>"$OUT/stderr_${arm}_${i}.log" \
      || echo "arm=$arm start=$i exit=$?" | tee -a "$OUT/meta.txt"
  done
done
echo "done $(date -u +%T)" | tee -a "$OUT/meta.txt"
echo "$OUT"
