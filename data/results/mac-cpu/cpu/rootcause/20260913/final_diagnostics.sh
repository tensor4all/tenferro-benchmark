#!/bin/bash
set -euo pipefail
source scripts/benchmark_host_idle.sh
assert_benchmark_host_idle
target/release/examples/rootcause_graph > /tmp/tenferro-rootcause/graphs.txt
for t in 1 4; do
 export OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t VECLIB_MAXIMUM_THREADS=$t VECLIB_NUM_THREADS=$t RAYON_NUM_THREADS=$t
 multi=false; [[ $t == 4 ]] && multi=true
 export XLA_FLAGS="--xla_cpu_multi_thread_eigen=$multi intra_op_parallelism_threads=$t"
 for rep in 0 1 2; do
  assert_benchmark_host_idle
  PJRT_NPROC=$t BENCH_RUNS=51 BENCH_WARMUPS=10 PUBLIC_API_SUITE_FILTER=cpu/elementwise_reduction PUBLIC_API_BENCHMARK_FILTER=tanh,reduce_max_axis0 .venv/bin/python scripts/benchmark_cpu_public_api_jax.py --num-threads $t --output /tmp/tenferro-rootcause/jax_exact_t${t}_rep${rep}.csv
 done
done
