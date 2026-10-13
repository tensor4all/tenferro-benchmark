#!/usr/bin/env bash
set -euo pipefail

# Run inside the repository CUDA devcontainer. Every timing process is sequential.
cd "$(dirname "$0")/.."
[[ -f /.dockerenv && -d /usr/local/cuda ]] || {
    echo 'Use devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json' >&2
    exit 1
}
family="${1:-fft}"
# The public API collection reuses family identities and never overlaps runners.
if [[ "$family" == all ]]; then
    case "${2:-}" in ''|--verify-only) ;; *) echo "Usage: $0 all [--verify-only]" >&2; exit 1;; esac
    [[ $# -le 2 ]] || { echo "Too many arguments" >&2; exit 1; }
    for member in fft nn isolated_ops structural linalg_public_api reuse_contraction lstsq; do
        "$0" "$member" "${@:2}"
    done
    exit 0
fi
case "$family" in fft|nn|isolated_ops|structural|linalg_public_api|reuse_contraction|lstsq) ;; *) echo "Unknown family: $family" >&2; exit 1;; esac
export GPU_API_VERIFY_ONLY=0
case "${2:-}" in
    '') default_device=0; export GPU_API_ROLE=benchmark ;;
    --verify-only) default_device=1; export GPU_API_ROLE=debug GPU_API_VERIFY_ONLY=1 ;;
    *) echo "Usage: $0 <family> [--verify-only]" >&2; exit 1 ;;
esac
export BENCHMARK_TARGET_PROFILE=nvidia-gpu
# Select the physical nvidia-smi index, then expose only that UUID. CUDA ordinal
# zero now means the same card in Rust, PyTorch and JAX, regardless of PCI order.
export GPU_BENCH_PHYSICAL_DEVICE="${GPU_BENCH_DEVICE:-$default_device}"
export GPU_BENCH_DEVICE_UUID
GPU_BENCH_DEVICE_UUID="$(nvidia-smi -i "$GPU_BENCH_PHYSICAL_DEVICE" --query-gpu=uuid --format=csv,noheader)"
[[ "$GPU_BENCH_DEVICE_UUID" == GPU-* && "$GPU_BENCH_DEVICE_UUID" != *$'\n'* ]] || exit 1
export CUDA_VISIBLE_DEVICES="$GPU_BENCH_DEVICE_UUID" GPU_BENCH_DEVICE=0
echo "$GPU_API_ROLE: physical GPU $GPU_BENCH_PHYSICAL_DEVICE ($GPU_BENCH_DEVICE_UUID), process CUDA ordinal 0"
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
if [[ "$GPU_API_VERIFY_ONLY" == 1 ]]; then
    run_dir="data/verification/nvidia-gpu/gpu/$family/$timestamp"
else
    run_dir="data/results/nvidia-gpu/gpu/$family/$timestamp"
fi
mkdir -p "$run_dir"
suite="data/instances/gpu_$family.json"
uv run python scripts/validate_benchmark_suite.py "benchmarks/gpu/$family.yaml"
# Always rebuild: do not measure a stale GPU binary.
cargo build --release --features cuda --bin benchmark_gpu_public_api -j "${CARGO_BUILD_JOBS:-4}"
uv run python scripts/collect_run_metadata.py \
    --suite-id "gpu/$family" --target-profile nvidia-gpu \
    --suite-file "benchmarks/gpu/$family.yaml" \
    --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --tenferro-dir extern/tenferro-rs \
    --features cuda --blas none --cuda-device-ordinal "$GPU_BENCH_PHYSICAL_DEVICE" \
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
if [[ "$GPU_API_VERIFY_ONLY" == 1 ]]; then
    uv run python scripts/format_gpu_public_api_results.py --suite "$suite" \
        --run-metadata "$run_dir/run.yaml" --verify-only \
        --output "$run_dir/verification.json" "${outputs[@]}"
else
    uv run python scripts/format_gpu_public_api_results.py --suite "$suite" \
        --run-metadata "$run_dir/run.yaml" --output "result/nvidia-gpu/gpu/$family.md" "${outputs[@]}"
fi
