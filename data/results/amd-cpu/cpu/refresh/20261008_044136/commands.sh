#!/usr/bin/env bash
set -euo pipefail
export TENFERRO_CPU_FEATURES=system-mkl
export PUBLICATION_GATE_FEATURES=system-mkl
export TENFERRO_CPU_BACKEND_KIND=blas
export BENCHMARK_TARGET_PROFILE=amd-cpu
export CARGO_TARGET_DIR="$PWD/target/compatibility-mkl"
export CARGO_BUILD_JOBS=8
export PUBLICATION_GATE_PROFILE=full
export RUN_PUBLIC_API_PROFILE=full
export BENCH_RUNS=15
export BENCH_WARMUPS=3
export BENCH_COVERAGE=full
export BENCH_EFFORT=standard
export FFT_BENCH_LENGTHS=1024,1048576
export PERMUTATION_EXTRA_FEATURES=hptt
export JULIA_NUM_PRECOMPILE_TASKS=4
export UV_NO_SYNC=1
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu
unset BENCH_INSTANCE BENCHMARK_ALLOW_BUSY_HOST
source scripts/benchmark_host_idle.sh
assert_benchmark_host_idle
./scripts/run_all.sh 1 4
./scripts/run_perf_issues.sh 1 4
