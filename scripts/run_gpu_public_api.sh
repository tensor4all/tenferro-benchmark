#!/usr/bin/env bash
set -euo pipefail

# Run inside the repository CUDA devcontainer (recommended) or on a host with a
# CUDA toolkit and driver. Every timing process is sequential.
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gpu_preflight.sh
source scripts/gpu_preflight.sh
# shellcheck source=scripts/gpu_arm_launch.sh
source scripts/gpu_arm_launch.sh
family="${1:-fft}"
# The public API collection reuses family identities and never overlaps runners.
if [[ "$family" == all ]]; then
    case "${2:-}" in ''|--verify-only) ;; *) echo "Usage: $0 all [--verify-only]" >&2; exit 1;; esac
    [[ $# -le 2 ]] || { echo "Too many arguments" >&2; exit 1; }
    for member in fft nn isolated_ops structural linalg_public_api_throughput linalg_public_api_latency reuse_contraction lstsq; do
        "$0" "$member" "${@:2}"
    done
    exit 0
fi
case "$family" in fft|nn|isolated_ops|structural|linalg_public_api_throughput|linalg_public_api_latency|reuse_contraction|lstsq) ;; *) echo "Unknown family: $family" >&2; exit 1;; esac
export GPU_API_VERIFY_ONLY=0
case "${2:-}" in
    '') default_device=0; export GPU_API_ROLE=benchmark ;;
    --verify-only) default_device=1; export GPU_API_ROLE=debug GPU_API_VERIFY_ONLY=1 ;;
    *) echo "Usage: $0 <family> [--verify-only]" >&2; exit 1 ;;
esac
export BENCHMARK_TARGET_PROFILE=nvidia-gpu
# Select the physical nvidia-smi index, then expose only that UUID. CUDA ordinal
# zero now means the same card in Rust, PyTorch and JAX, regardless of PCI order.
configure_gpu_bench_env "$default_device"
echo "$GPU_API_ROLE: physical GPU $GPU_BENCH_PHYSICAL_DEVICE ($GPU_BENCH_DEVICE_UUID), process CUDA ordinal 0"
export BENCH_RUNS="${BENCH_RUNS:-15}" BENCH_WARMUPS="${BENCH_WARMUPS:-3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUBECL_DEBUG_LOG=0 RUST_MIN_STACK=67108864
# As in run_gpu_permutation.sh: only take the devcontainer path when it exists,
# so the runner's DEFAULT_CUTENSOR_PATHS fallback still finds a cuTENSOR that is
# on LD_LIBRARY_PATH.
CUTENSOR_DEVCONTAINER_PATH="/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2"
if [[ -z "${TENFERRO_CUTENSOR_PATH:-}" && -f "$CUTENSOR_DEVCONTAINER_PATH" ]]; then
    export TENFERRO_CUTENSOR_PATH="$CUTENSOR_DEVCONTAINER_PATH"
fi
TENFERRO_CPU_FEATURES=native bash scripts/setup_extern_deps.sh

# Preserve pinned checkouts; follow main only when the user has selected main.
tenferro_dir="$PWD/extern/tenferro-rs"
update_tenferro_checkout_if_main "$tenferro_dir"
# The routine selection hands in a materialized instance document and suite
# declaration, so one family can be collected with a curated case set.
timestamp="${BENCHMARK_TIMESTAMP:-${GPU_BENCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}}"
if [[ "$GPU_API_VERIFY_ONLY" == 1 ]]; then
    run_dir="data/verification/nvidia-gpu/gpu/$family/$timestamp"
else
    run_dir="data/results/nvidia-gpu/gpu/$family/$timestamp"
fi
mkdir -p "$run_dir"
declaration="${GPU_API_SUITE_DECL:-benchmarks/gpu/$family.yaml}"
uv run python scripts/validate_benchmark_suite.py "$declaration"
# A family's cases are its declaration's `problems.include` over its instance
# document, so the document handed to the collectors is derived from the
# declaration: one shape class per page cannot drift into measuring the other's
# cases. The routine path already passes its own curated document.
if [[ -n "${GPU_API_SUITE_FILE:-}" ]]; then
    suite="$GPU_API_SUITE_FILE"
else
    suite="$run_dir/instances.json"
    uv run python scripts/gpu_routine_selection.py --instances "$declaration" --output "$suite"
fi
# The run keeps the selection it measured, so its page and sidecar can be
# rendered from the run directory alone.
if [[ "$suite" != "$run_dir/instances.json" ]]; then
    cp "$suite" "$run_dir/instances.json"
fi
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
    --tenferro-dir "$tenferro_dir" \
    --features cuda --blas none --cuda-device-ordinal "$GPU_BENCH_PHYSICAL_DEVICE" \
    --output "$run_dir/run.yaml"
# Record this run's own copy rate: time one plain device copy of a known size on
# this device and put the traffic rate it reached in run.yaml. Every record below
# carries its logical bytes against that rate as a diagnostic. A failed or skipped
# calibration costs the ratio, never the run.
if [[ "$GPU_API_VERIFY_ONLY" != 1 ]]; then
    uv run python scripts/gpu_copy_calibration.py --run-metadata "$run_dir/run.yaml" \
        --device "$GPU_BENCH_DEVICE" \
        || echo "device copy calibration failed; the records will carry no copy ratio" >&2
fi
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
# Each arm gets a record under $run_dir/arms/: the CPU it ran on and that CPU's busy
# fraction (GPU_BENCH_CPU_SELECT=1), or why it was skipped. An arm whose output an
# earlier process of the same run stamp already wrote is skipped explicitly instead of
# being started and recorded as a failure (#335).
run_arm() {
    local backend="$1" output="$2" rc=0
    shift 2
    gpu_arm_prepare "$run_dir" "$backend" "$output" || rc=$?
    if [[ "$rc" == "$GPU_ARM_SKIPPED" ]]; then
        outputs+=("$output")
        return 0
    fi
    [[ "$rc" == 0 ]] || gpu_arm_fail_unless_runnable "$rc"
    "${GPU_ARM_PREFIX[@]+"${GPU_ARM_PREFIX[@]}"}" "$@" || rc=$?
    gpu_arm_finish "$run_dir" "$backend" "$rc"
    [[ "$rc" == 0 ]] || exit "$rc"
    outputs+=("$output")
}
for backend in ${rust_arms[@]+"${rust_arms[@]}"}; do
    output="$run_dir/$backend.jsonl"
    run_arm "$backend" "$output" "${CARGO_TARGET_DIR:-target}/release/benchmark_gpu_public_api" \
        --fixtures "$run_dir/fixtures/manifest.json" --backend "$backend" --output "$output"
done
for backend in ${python_arms[@]+"${python_arms[@]}"}; do
    output="$run_dir/$backend.jsonl"
    run_arm "$backend" "$output" uv run python scripts/benchmark_gpu_public_api.py --suite "$suite" \
        --fixtures "$run_dir/fixtures" --backend "$backend" --output "$output"
done
# The arm records go into run.yaml, so the page's run metadata names the CPU each arm ran on.
uv run python scripts/gpu_measurement_cpu.py record-run --run-metadata "$run_dir/run.yaml" \
    "$run_dir"/arms/*.json
if [[ "$GPU_API_VERIFY_ONLY" == 1 ]]; then
    uv run python scripts/format_gpu_public_api_results.py --suite "$suite" \
        --run-metadata "$run_dir/run.yaml" --verify-only \
        --output "$run_dir/verification.json" "${outputs[@]+"${outputs[@]}"}"
    exit 0
fi
# Record the diagnostic block on every record of the family: the logical bytes the
# case must move per operation and the ratio of its implied rate to this run's
# measured device copy rate. Recording never changes a status, a timing or the exit
# code, and a ratio above the copy rate is legitimate for a cache-resident case.
for output in "${outputs[@]}"; do
    uv run python scripts/gpu_bandwidth_guard.py --run-metadata "$run_dir/run.yaml" "$output" \
        || echo "could not record the copy ratio for $output" >&2
done
# The routine path reads one record contract: fold this family's rows and build
# its summary from them, then publish the family page next to those artifacts.
uv run python scripts/normalize_gpu_records.py --output "$run_dir/records.jsonl" \
    --summary "$run_dir/report.summary.json" --run-metadata "$run_dir/run.yaml" "${outputs[@]+"${outputs[@]}"}"
uv run python scripts/format_gpu_public_api_results.py --suite "$suite" \
    --run-metadata "$run_dir/run.yaml" --output "$run_dir/report.md" "${outputs[@]+"${outputs[@]}"}"
if [[ "${GPU_BENCH_SKIP_LATEST:-0}" != 1 ]]; then
    mkdir -p result/nvidia-gpu/gpu
    cp "$run_dir/report.md" "result/nvidia-gpu/gpu/$family.md"
    cp "$run_dir/report.summary.json" "result/nvidia-gpu/gpu/$family.summary.json"
fi
