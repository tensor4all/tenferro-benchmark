#!/usr/bin/env bash
# The GPU preflight selects a toolkit and one physical device on a container or
# on a host; every failed check names itself. Fake nvidia-smi/nvcc on PATH.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Only the tools the preflight calls; nvcc and nvidia-smi are added per case so
# a host that has them (this repository's Linux devcontainers do) cannot leak in.
FARM="$TMP/farm"
mkdir -p "$FARM"
for tool in bash dirname readlink cat; do ln -sf "$(command -v "$tool")" "$FARM/$tool"; done

SMITOOL="$TMP/smitool"
mkdir -p "$SMITOOL"
cat > "$SMITOOL/nvidia-smi" <<'SH'
#!/usr/bin/env bash
# The device UUID follows the requested physical index, so a dropped -i shows up.
index=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i) index="$2"; shift 2 ;;
        *) shift ;;
    esac
done
case "${FAKE_SMI_MODE:-ok}" in
    fail) echo "NVML: driver not loaded" >&2; exit 9 ;;
    bad) echo "not-a-uuid"; exit 0 ;;
    multi) printf 'GPU-aaa\nGPU-bbb\n'; exit 0 ;;
esac
case "$index" in
    0) printf 'GPU-1234abcd\n' ;;
    1) printf 'GPU-5678efgh\n' ;;
    *)
        echo "No devices were found for index $index" >&2
        exit 6
        ;;
esac
SH
chmod +x "$SMITOOL/nvidia-smi"

REAL_CUDA="$TMP/real-cuda"
mkdir -p "$REAL_CUDA/bin" "$REAL_CUDA/lib64"
printf '#!/usr/bin/env bash\necho "Cuda compilation tools, release 12.4, V12.4.131"\n' > "$REAL_CUDA/bin/nvcc"
chmod +x "$REAL_CUDA/bin/nvcc"
NVCC_LINK="$TMP/nvcclink"
mkdir -p "$NVCC_LINK"
ln -sf "$REAL_CUDA/bin/nvcc" "$NVCC_LINK/nvcc"

real() { (cd "$1" && pwd -P); }

# Runs configure_gpu_bench_env in a clean environment and prints the selected
# values: toolkit root, CUDA_PATH, physical index, CUDA ordinal, UUID, loader path.
preflight() {
    local path="$1" default="$2"
    shift 2
    env -i PATH="$path" HOME="$TMP" "$@" /bin/bash -c '
        source ./scripts/gpu_preflight.sh
        configure_gpu_bench_env "$1" >/dev/null || exit $?
        printf "%s|%s|%s|%s|%s|%s\n" \
            "$CUDA_HOME" "$CUDA_PATH" "$GPU_BENCH_PHYSICAL_DEVICE" "$GPU_BENCH_DEVICE" \
            "$GPU_BENCH_DEVICE_UUID" "${LD_LIBRARY_PATH:-}"
    ' _ "$default"
}

expect_failure() {
    local name="$1" needle="$2"
    shift 2
    local status=0
    preflight "$@" >"$TMP/out" 2>"$TMP/err" || status=$?
    if [[ "$status" == 0 ]]; then
        echo "$name: expected a failure, got success" >&2
        exit 1
    fi
    if ! grep -qi -- "$needle" "$TMP/err"; then
        echo "$name: error did not mention '$needle'" >&2
        cat "$TMP/err" >&2
        exit 1
    fi
}

# --- toolkit selection -------------------------------------------------------
EXPECTED="$(real "$REAL_CUDA")"
line="$(preflight "$FARM:$SMITOOL" 0 CUDA_HOME="$REAL_CUDA" FAKE_SMI_MODE=ok)"
[[ "$line" == "$EXPECTED|$EXPECTED|0|0|GPU-1234abcd|$EXPECTED/lib64" ]] \
    || { echo "CUDA_HOME selection: $line" >&2; exit 1; }

line="$(preflight "$FARM:$SMITOOL" 0 CUDA_PATH="$REAL_CUDA" FAKE_SMI_MODE=ok)"
[[ "$line" == "$EXPECTED|$EXPECTED|0|0|GPU-1234abcd|$EXPECTED/lib64" ]] \
    || { echo "CUDA_PATH selection: $line" >&2; exit 1; }

# A module/alternatives install: nvcc on PATH is a symlink into the toolkit.
line="$(preflight "$FARM:$SMITOOL:$NVCC_LINK" 0 FAKE_SMI_MODE=ok)"
[[ "$line" == "$EXPECTED|$EXPECTED|0|0|GPU-1234abcd|$EXPECTED/lib64" ]] \
    || { echo "nvcc-on-PATH selection: $line" >&2; exit 1; }

line="$(preflight "$FARM:$SMITOOL" 0 CUDA_HOME="$REAL_CUDA" LD_LIBRARY_PATH=/opt/keep FAKE_SMI_MODE=ok)"
[[ "$line" == *"|$EXPECTED/lib64:/opt/keep" ]] \
    || { echo "existing LD_LIBRARY_PATH not preserved: $line" >&2; exit 1; }

expect_failure "no toolkit" "CUDA_HOME" "$FARM" 0
expect_failure "bad CUDA_HOME" "$TMP/missing" "$FARM" 0 CUDA_HOME="$TMP/missing"

# --- required checks fail with their own message ------------------------------
expect_failure "no nvidia-smi" "nvidia-smi" "$FARM" 0 CUDA_HOME="$REAL_CUDA"
expect_failure "failing nvidia-smi" "NVML" "$FARM:$SMITOOL" 0 CUDA_HOME="$REAL_CUDA" FAKE_SMI_MODE=fail
expect_failure "non-UUID answer" "no single GPU UUID" "$FARM:$SMITOOL" 0 CUDA_HOME="$REAL_CUDA" FAKE_SMI_MODE=bad
expect_failure "multi-GPU answer" "no single GPU UUID" "$FARM:$SMITOOL" 0 CUDA_HOME="$REAL_CUDA" FAKE_SMI_MODE=multi
# A GPU the job was not allocated is not silently substituted.
expect_failure "unallocated index" "No devices were found for index 7" \
    "$FARM:$SMITOOL" 0 CUDA_HOME="$REAL_CUDA" GPU_BENCH_DEVICE=7

# --- physical device defaults are preserved ---------------------------------
line="$(preflight "$FARM:$SMITOOL" 0 CUDA_HOME="$REAL_CUDA" GPU_BENCH_DEVICE=1 FAKE_SMI_MODE=ok)"
[[ "$line" == *"|1|0|GPU-5678efgh|"* ]] || { echo "GPU_BENCH_DEVICE=1: $line" >&2; exit 1; }
line="$(preflight "$FARM:$SMITOOL" 1 CUDA_HOME="$REAL_CUDA" FAKE_SMI_MODE=ok)"
[[ "$line" == *"|1|0|GPU-5678efgh|"* ]] || { echo "default physical device 1: $line" >&2; exit 1; }
line="$(preflight "$FARM:$SMITOOL" 1 CUDA_HOME="$REAL_CUDA" GPU_BENCH_DEVICE=0 FAKE_SMI_MODE=ok)"
[[ "$line" == *"|0|0|GPU-1234abcd|"* ]] || { echo "explicit device 0 over default: $line" >&2; exit 1; }

# --- the routine GPU entry point honors the same offline opt-out ------------
grep -q 'update_tenferro_checkout_if_main' scripts/run_gpu_suite.sh
if grep -q 'pull --ff-only' scripts/run_gpu_suite.sh; then
    echo "run_gpu_suite.sh pulls on its own instead of through the helper" >&2
    exit 1
fi

# --- the four collectors no longer gate on a container ----------------------
for script in scripts/run_gpu_linalg_jvp_vjp.sh scripts/run_gpu_public_api.sh \
              scripts/run_gpu_throughput.sh scripts/run_cpu_followup.sh; do
    bash -n "$script"
    if grep -q 'dockerenv' "$script"; then
        echo "$script still refuses to run outside a container" >&2
        exit 1
    fi
done
grep -q 'configure_gpu_bench_env' scripts/run_gpu_linalg_jvp_vjp.sh
grep -q 'configure_gpu_bench_env' scripts/run_gpu_public_api.sh
grep -q 'configure_gpu_bench_env' scripts/run_gpu_throughput.sh

echo "gpu preflight ok"
