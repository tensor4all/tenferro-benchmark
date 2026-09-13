#!/usr/bin/env bash
set -euo pipefail
cd /Users/hiroshi/projects/tensor4all/tenferro-benchmark-m5-refresh
export PATH="/Users/hiroshi/.juliaup/bin:$PATH"
export BENCHMARK_COMMIT="$(git rev-parse HEAD)"
export BENCHMARK_TARGET_PROFILE=mac-cpu
export TENFERRO_CPU_FEATURES=system-accelerate
export PUBLICATION_GATE_PROFILE=full
export FFT_BENCH_LENGTHS=1024,16384,1048576
export RUN_PERMUTATION_SUITE=1
source scripts/benchmark_host_idle.sh
assert_benchmark_host_idle
./scripts/run_all.sh 1 4
