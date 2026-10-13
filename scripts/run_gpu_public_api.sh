#!/usr/bin/env bash
set -euo pipefail

# Run inside the repository CUDA devcontainer. Every timing process is sequential.
cd "$(dirname "$0")/.."
[[ -f /.dockerenv && -d /usr/local/cuda ]] || {
    echo 'Use devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json' >&2
    exit 1
}
family="${1:-fft}"
case "$family" in fft|nn|isolated_ops|structural|linalg_public_api|reuse_contraction) ;; *) echo "Unknown family: $family" >&2; exit 1;; esac
export BENCHMARK_TARGET_PROFILE=nvidia-gpu
export GPU_BENCH_DEVICE="${GPU_BENCH_DEVICE:-0}"
export BENCH_RUNS="${BENCH_RUNS:-15}" BENCH_WARMUPS="${BENCH_WARMUPS:-3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUBECL_DEBUG_LOG=0 CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" RUST_MIN_STACK=67108864
export TENFERRO_CUTENSOR_PATH="${TENFERRO_CUTENSOR_PATH:-/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2}"
export LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TENFERRO_CPU_FEATURES=native bash scripts/setup_extern_deps.sh

# Preserve pinned checkouts; follow main only when the user has selected main.
if [[ "$(git -C extern/tenferro-rs branch --show-current)" == main ]]; then
    # The CUDA image has git but no SSH client; public GitHub fetches need no key.
    git -C extern/tenferro-rs -c 'url.https://github.com/.insteadOf=git@github.com:' pull --ff-only
fi
timestamp="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
run_dir="data/results/nvidia-gpu/gpu/$family/$timestamp"
mkdir -p "$run_dir"
suite="data/instances/gpu_$family.json"
uv run python scripts/validate_benchmark_suite.py "benchmarks/gpu/$family.yaml"
# Always rebuild: do not measure a stale GPU binary.
cargo build --release --features cuda --bin benchmark_gpu_public_api -j "${CARGO_BUILD_JOBS:-4}"
uv run python scripts/collect_run_metadata.py \
    --suite-id "gpu/$family" --target-profile nvidia-gpu \
    --suite-file "benchmarks/gpu/$family.yaml" \
    --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --tenferro-dir extern/tenferro-rs \
    --features cuda --blas none --cuda-device-ordinal "$GPU_BENCH_DEVICE" \
    --output "$run_dir/run.yaml"
uv run python scripts/benchmark_gpu_public_api.py --suite "$suite" \
    --fixtures "$run_dir/fixtures" --backend prepare
outputs=()
for backend in tenferro-cuda-direct tenferro-cuda-eager; do
    output="$run_dir/$backend.jsonl"
    "${CARGO_TARGET_DIR:-target}/release/benchmark_gpu_public_api" \
        --fixtures "$run_dir/fixtures/manifest.json" --backend "$backend" --output "$output"
    outputs+=("$output")
done
for backend in pytorch-cuda jax-cuda jax-cuda-jit; do
    output="$run_dir/$backend.jsonl"
    uv run python scripts/benchmark_gpu_public_api.py --suite "$suite" \
        --fixtures "$run_dir/fixtures" --backend "$backend" --output "$output"
    outputs+=("$output")
done
uv run python scripts/format_gpu_public_api_results.py --suite "$suite" \
    --run-metadata "$run_dir/run.yaml" --output "result/nvidia-gpu/gpu/$family.md" "${outputs[@]}"
