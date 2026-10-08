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
export BENCHMARK_COMMIT="$(git rev-parse HEAD)"
printf '%s\n' "$PWD/data/results/amd-cpu/cpu/einsum/20261008_044146" "$PWD/data/results/amd-cpu/cpu/einsum/20261008_053424" > data/results/amd-cpu/cpu/refresh/20261008_044136/runs-manifest.txt
uv run python scripts/format_cpu_thread_reports.py --runs-manifest data/results/amd-cpu/cpu/refresh/20261008_044136/runs-manifest.txt --root .
./scripts/run_cpu_fft.sh 1 4
./scripts/run_cpu_public_api.sh 1 4
./scripts/run_small_work.sh 1 4
./scripts/run_cpu_session.sh 1 4
./scripts/run_route_diagnostic.sh 1 4
./scripts/run_permutation.sh 1 4
./scripts/run_perf_issues.sh 1 4
