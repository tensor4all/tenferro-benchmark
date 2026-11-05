#!/usr/bin/env bash

# GPU collection preflight shared by the CUDA collectors.
#
# A container is not required. The devcontainer is the recommended environment,
# and whether a run is comparable is decided by the recorded run metadata
# (container, toolkit, driver), not by refusing to start outside /.dockerenv.
#
# Sourced, not executed. See docs/gpu-devcontainer.md, "Host (non-container) runs".

# shellcheck source=scripts/tenferro_checkout.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tenferro_checkout.sh"

# Toolkit root in the order the collectors use it: CUDA_HOME, CUDA_PATH, then the
# nvcc on PATH (a `module load cuda` sets PATH and LD_LIBRARY_PATH only). The
# root is never assumed to be /usr/local/cuda.
resolve_cuda_toolkit_root() {
    local candidate
    for candidate in "${CUDA_HOME:-}" "${CUDA_PATH:-}"; do
        [[ -z "$candidate" ]] && continue
        if [[ ! -d "$candidate" ]]; then
            echo "ERROR: CUDA toolkit directory $candidate (CUDA_HOME/CUDA_PATH) does not exist." >&2
            return 1
        fi
        printf '%s\n' "$candidate"
        return 0
    done
    if command -v nvcc >/dev/null 2>&1; then
        local nvcc nvcc_path
        nvcc="$(command -v nvcc)"
        nvcc_path="$nvcc"
        # A module/alternatives install symlinks nvcc outside the toolkit tree.
        readlink -f "$nvcc" >/dev/null 2>&1 && nvcc_path="$(readlink -f "$nvcc")"
        (cd "$(dirname "$nvcc_path")/.." && pwd -P)
        return 0
    fi
    cat >&2 <<'EOF'
ERROR: no CUDA toolkit found: CUDA_HOME and CUDA_PATH are unset and no nvcc is
on PATH. Set CUDA_HOME to the toolkit root, or load the CUDA module / add its
bin directory (containing nvcc) to PATH.
EOF
    return 1
}

# Export the persistent NVRTC artifact cache every Rust arm shares.
#
# A NVRTC-compiled module costs about 48 s the first time a process compiles it,
# and the collectors start one process per backend per suite while each case
# builds its own engine or traced runtime. A persistent directory outside the
# repository lets a later process read the cubins back instead of compiling them
# again. It is keyed by the NVRTC version and the device's compute capability, so
# a toolkit upgrade or another card cannot read a stale entry. An explicit value,
# including `none`, wins and skips the discovery.
# Usage: configure_cuda_artifact_cache [physical nvidia-smi index]
configure_cuda_artifact_cache() {
    local physical_device="${1:-${GPU_BENCH_PHYSICAL_DEVICE:-0}}"
    local gpu_arch=unknown capability nvrtc_lib nvrtc_version
    if [[ -n "${TENFERRO_CUDA_ARTIFACT_CACHE:-}" ]]; then
        echo "CUDA artifacts: $TENFERRO_CUDA_ARTIFACT_CACHE"
        return 0
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        # A driver that does not know compute_cap must not take the runner down.
        capability="$(nvidia-smi -i "$physical_device" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '. \n' | head -1 || true)"
        if [[ -n "$capability" ]]; then gpu_arch="sm_$capability"; fi
    fi
    nvrtc_lib="$(ls "${CUDA_HOME:-/usr/local/cuda}"/lib64/libnvrtc.so.* 2>/dev/null | sort -V | tail -1 || true)"
    nvrtc_version="${nvrtc_lib##*/libnvrtc.so.}"
    # No cache root to name means the memory-only behaviour, not an abort: HOME is
    # unset in some batch environments and this runs under `set -u`.
    local cache_root="${XDG_CACHE_HOME:-${HOME:+$HOME/.cache}}"
    if [[ -z "$cache_root" ]]; then
        return 0
    fi
    export TENFERRO_CUDA_ARTIFACT_CACHE="$cache_root/tenferro-benchmark/cuda-artifacts/${nvrtc_version:-unknown}/$gpu_arch"
    echo "CUDA artifacts: $TENFERRO_CUDA_ARTIFACT_CACHE"
}

# Export one toolkit root for builds, runtimes and metadata, then expose exactly
# the selected physical GPU as process-local CUDA ordinal zero.
# Usage: configure_gpu_bench_env [default physical nvidia-smi index]
configure_gpu_bench_env() {
    local default_physical_device="${1:-0}"
    local toolkit_root uuid
    toolkit_root="$(resolve_cuda_toolkit_root)" || return 1
    export CUDA_HOME="$toolkit_root"
    export CUDA_PATH="$toolkit_root"
    export LD_LIBRARY_PATH="$toolkit_root/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi not found on PATH; the NVIDIA driver tools are missing." >&2
        return 1
    fi
    export GPU_BENCH_PHYSICAL_DEVICE="${GPU_BENCH_DEVICE:-$default_physical_device}"
    if ! uuid="$(nvidia-smi -i "$GPU_BENCH_PHYSICAL_DEVICE" --query-gpu=uuid --format=csv,noheader 2>&1)"; then
        echo "ERROR: nvidia-smi could not read physical GPU $GPU_BENCH_PHYSICAL_DEVICE:" >&2
        echo "$uuid" >&2
        return 1
    fi
    if [[ "$uuid" != GPU-* || "$uuid" == *$'\n'* ]]; then
        echo "ERROR: nvidia-smi returned no single GPU UUID for physical GPU $GPU_BENCH_PHYSICAL_DEVICE: '$uuid'" >&2
        echo "Select the card with GPU_BENCH_DEVICE (physical nvidia-smi index, see nvidia-smi -L)." >&2
        return 1
    fi
    export GPU_BENCH_DEVICE_UUID="$uuid"
    export CUDA_VISIBLE_DEVICES="$uuid"
    export GPU_BENCH_DEVICE=0
    configure_cuda_artifact_cache "$GPU_BENCH_PHYSICAL_DEVICE"
}
