#!/bin/bash
set -euo pipefail
source scripts/benchmark_host_idle.sh
for t in 1 4; do
 assert_benchmark_host_idle
 multi=false; [[ $t == 4 ]] && multi=true
 OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t VECLIB_MAXIMUM_THREADS=$t VECLIB_NUM_THREADS=$t XLA_FLAGS="--xla_cpu_multi_thread_eigen=$multi intra_op_parallelism_threads=$t" .venv/bin/python /tmp/tenferro-rootcause/jax_threads.py > /tmp/tenferro-rootcause/jax_threads_t$t.jsonl
done
