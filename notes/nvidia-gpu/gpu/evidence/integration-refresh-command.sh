#!/usr/bin/env bash
set -euo pipefail
# Run from the benchmark worktree inside bench-gpu-20260919.
source scripts/thread_env.sh
configure_cpu_thread_env 1
export CARGO_TARGET_DIR="$PWD/target" CARGO_BUILD_JOBS=16
export CUDA_PATH=/usr/local/cuda CUBECL_DEBUG_LOG=0 RUST_MIN_STACK=67108864
export BENCHMARK_TARGET_PROFILE=nvidia-gpu
export GPU_BENCH_TIMESTAMP=20260919_142600
export BENCHMARK_TIMESTAMP="$GPU_BENCH_TIMESTAMP"
for suite in dense einsum elementwise sparse tensornetwork linalg_ad_latency linalg_jvp_vjp permutation; do
    test ! -e "data/results/nvidia-gpu/gpu/$suite/$GPU_BENCH_TIMESTAMP"
done
print_cpu_thread_env
git rev-parse HEAD
git -C extern/tenferro-rs rev-parse HEAD
git -C extern/tenferro-rs status --porcelain
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
GPU_BENCH_SUITE=benchmarks/gpu/dense.yaml,benchmarks/gpu/einsum.yaml,benchmarks/gpu/elementwise.yaml,benchmarks/gpu/sparse.yaml,benchmarks/gpu/tensornetwork.yaml bash scripts/run_gpu_suite.sh
bash scripts/run_gpu_linalg_ad_latency.sh
bash scripts/run_gpu_linalg_jvp_vjp.sh
bash scripts/run_gpu_permutation.sh
sha256sum target/release/benchmark_gpu_rust target/release/benchmark_gpu_linalg_ad target/release/benchmark_gpu_permutation
