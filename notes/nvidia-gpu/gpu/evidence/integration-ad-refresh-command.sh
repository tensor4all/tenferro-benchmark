#!/usr/bin/env bash
set -euo pipefail
trap 'echo $? > data/diagnostics/integration-ad-refresh.exit' EXIT
source scripts/thread_env.sh
configure_cpu_thread_env 1
export CARGO_TARGET_DIR="$PWD/target" CARGO_BUILD_JOBS=16
export CUDA_PATH=/usr/local/cuda CUBECL_DEBUG_LOG=0 RUST_MIN_STACK=67108864
export BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_TIMESTAMP=20260919_151433
for suite in linalg_ad_latency linalg_jvp_vjp; do
    test ! -e "data/results/nvidia-gpu/gpu/$suite/$GPU_BENCH_TIMESTAMP"
done
print_cpu_thread_env
git rev-parse HEAD
git -C extern/tenferro-rs rev-parse HEAD
bash scripts/run_gpu_linalg_ad_latency.sh
bash scripts/run_gpu_linalg_jvp_vjp.sh
sha256sum target/release/benchmark_gpu_linalg_ad
