#!/usr/bin/env bash
set -euo pipefail

# Run inside the repository CUDA devcontainer. Every timing process is sequential.
cd "$(dirname "$0")/.."
[[ -f /.dockerenv && -d "${CUDA_HOME:-/usr/local/cuda}" ]] || {
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
# The routine selection hands in a materialized instance document and suite
# declaration, so one family can be collected with a curated case set.
timestamp="${BENCHMARK_TIMESTAMP:-${GPU_BENCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}}"
if [[ "$GPU_API_VERIFY_ONLY" == 1 ]]; then
    run_dir="data/verification/nvidia-gpu/gpu/$family/$timestamp"
else
    run_dir="data/results/nvidia-gpu/gpu/$family/$timestamp"
fi
mkdir -p "$run_dir"
suite="${GPU_API_SUITE_FILE:-data/instances/gpu_$family.json}"
declaration="${GPU_API_SUITE_DECL:-benchmarks/gpu/$family.yaml}"
uv run python scripts/validate_benchmark_suite.py "$declaration"
if [[ "${GPU_BENCH_PREBUILT_RUST:-0}" == 1 ]]; then
    # The hosted build job ships this binary; the benchmark pod has no Rust toolchain.
    python3 scripts/ci/gpu_artifact.py "${CARGO_TARGET_DIR:-target}/release/benchmark_gpu_public_api"
else
    # Always rebuild: do not measure a stale GPU binary.
    cargo build --release --features cuda --bin benchmark_gpu_public_api -j "${CARGO_BUILD_JOBS:-4}"
fi
uv run python scripts/collect_run_metadata.py \
    --suite-id "gpu/$family" --target-profile nvidia-gpu \
    --suite-file "$declaration" \
    --timestamp "${GPU_BENCH_RUN_TIMESTAMP:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}" \
    --tenferro-dir extern/tenferro-rs \
    --features cuda --blas none --cuda-device-ordinal "$GPU_BENCH_PHYSICAL_DEVICE" \
    --output "$run_dir/run.yaml"
uv run python scripts/benchmark_gpu_public_api.py --suite "$suite" \
    --fixtures "$run_dir/fixtures" --backend prepare
# The routine run names the arms; without a filter the family's own arms all run.
mapfile -t api_backends < <(uv run python - "$suite" "${GPU_BENCH_BACKENDS:-}" <<'PY'
import json, sys
declared = json.load(open(sys.argv[1]))["backends"]
requested = [name for name in sys.argv[2].split(",") if name] or declared
for name in requested:
    if name in declared:
        print(name)
PY
)
[[ ${#api_backends[@]} -gt 0 ]] || { echo "No public API backends selected" >&2; exit 1; }
rust_arms=()
python_arms=()
for backend in "${api_backends[@]}"; do
    case "$backend" in
        tenferro-cuda-direct|tenferro-cuda-eager) rust_arms+=("$backend") ;;
        *) python_arms+=("$backend") ;;
    esac
done
outputs=()
for backend in ${rust_arms[@]+"${rust_arms[@]}"}; do
    output="$run_dir/$backend.jsonl"
    "${CARGO_TARGET_DIR:-target}/release/benchmark_gpu_public_api" \
        --fixtures "$run_dir/fixtures/manifest.json" --backend "$backend" --output "$output"
    outputs+=("$output")
done
for backend in ${python_arms[@]+"${python_arms[@]}"}; do
    output="$run_dir/$backend.jsonl"
    uv run python scripts/benchmark_gpu_public_api.py --suite "$suite" \
        --fixtures "$run_dir/fixtures" --backend "$backend" --output "$output"
    outputs+=("$output")
done
if [[ "$GPU_API_VERIFY_ONLY" == 1 ]]; then
    uv run python scripts/format_gpu_public_api_results.py --suite "$suite" \
        --run-metadata "$run_dir/run.yaml" --verify-only \
        --output "$run_dir/verification.json" "${outputs[@]}"
    exit 0
fi
# The routine path reads one record contract: fold this family's rows and build
# its summary from them, then publish the family page next to those artifacts.
uv run python scripts/normalize_gpu_records.py --output "$run_dir/records.jsonl" \
    --summary "$run_dir/report.summary.json" --run-metadata "$run_dir/run.yaml" "${outputs[@]}"
uv run python scripts/format_gpu_public_api_results.py --suite "$suite" \
    --run-metadata "$run_dir/run.yaml" --output "$run_dir/report.md" "${outputs[@]}"
if [[ "${GPU_BENCH_SKIP_LATEST:-0}" != 1 ]]; then
    mkdir -p result/nvidia-gpu/gpu
    cp "$run_dir/report.md" "result/nvidia-gpu/gpu/$family.md"
    cp "$run_dir/report.summary.json" "result/nvidia-gpu/gpu/$family.summary.json"
fi
