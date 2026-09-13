#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source scripts/thread_env.sh
source scripts/benchmark_host_idle.sh
export PUBLICATION_GATE_PROFILE=full TENFERRO_CPU_BACKEND_KIND=blas
for t in 1 4; do
 [[ ! -e data/results/mac-cpu/cpu/cpu_ops/20260913_130307/views_confirmed_t${t}.csv ]] || { echo "Refusing to append to an existing confirmation run" >&2; exit 1; }
 configure_cpu_thread_env "$t"
 assert_benchmark_host_idle
 PUBLIC_API_SUITE_FILTER=cpu/view_metadata PUBLIC_API_EXECUTION_FILTER=direct target/release/benchmark_cpu_public_api --num-threads "$t" --output data/results/mac-cpu/cpu/cpu_ops/20260913_130307/views_confirmed_t${t}.csv
done
