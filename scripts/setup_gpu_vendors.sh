#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Install vendor GPU libraries needed by the cutlass and ginkgo benchmark
# backends inside the CUDA devcontainer.
#
#   - CUTLASS (header-only) → $CUTLASS_DIR (default /opt/cutlass)
#   - Ginkgo (built with CUDA) → $GINKGO_DIR (default /opt/ginkgo)
#
# CUTLASS is a quick clone. Ginkgo is a full CUDA build (~10-15 min).
# Both backends degrade to `not_configured` records if their library is
# absent, so this script is optional unless cutlass/ginkgo columns are wanted.
#
# Usage:
#   ./scripts/setup_gpu_vendors.sh [cutlass|ginkgo|all]
# ---------------------------------------------------------------------------

TARGET="${1:-all}"

CUTLASS_DIR="${CUTLASS_DIR:-/opt/cutlass}"
CUTLASS_TAG="${CUTLASS_TAG:-v3.7.0}"
GINKGO_DIR="${GINKGO_DIR:-/opt/ginkgo}"
GINKGO_SRC="${GINKGO_SRC:-/opt/ginkgo-src}"
GINKGO_TAG="${GINKGO_TAG:-v1.8.0}"
CUDA_ARCH="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' || echo 86)}"

log() { printf '[setup_gpu_vendors] %s\n' "$*" >&2; }

ensure_writable() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        sudo mkdir -p "$dir"
        sudo chown "$(id -u):$(id -g)" "$dir"
    fi
}

setup_cutlass() {
    if [[ -f "$CUTLASS_DIR/include/cutlass/cutlass.h" ]]; then
        log "CUTLASS already present at $CUTLASS_DIR"
        return
    fi
    ensure_writable "$CUTLASS_DIR"
    log "Cloning CUTLASS $CUTLASS_TAG into $CUTLASS_DIR"
    git clone --depth=1 --branch "$CUTLASS_TAG" \
        https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
    log "CUTLASS ready (header-only)"
}

setup_ginkgo() {
    if [[ -f "$GINKGO_DIR/lib/libginkgo.so" ]]; then
        log "Ginkgo already installed at $GINKGO_DIR"
        return
    fi
    ensure_writable "$GINKGO_DIR"
    ensure_writable "$GINKGO_SRC"
    log "Cloning Ginkgo $GINKGO_TAG into $GINKGO_SRC"
    if [[ ! -d "$GINKGO_SRC/.git" ]]; then
        git clone --depth=1 --branch "$GINKGO_TAG" \
            https://github.com/ginkgo-project/ginkgo.git "$GINKGO_SRC"
    fi
    log "Configuring Ginkgo (CUDA arch=$CUDA_ARCH)"
    CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
    LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        cmake -S "$GINKGO_SRC" -B "$GINKGO_SRC/build" -GNinja \
            -DCMAKE_BUILD_TYPE=Release \
            -DGINKGO_BUILD_CUDA=ON \
            -DGINKGO_BUILD_OMP=OFF \
            -DGINKGO_BUILD_REFERENCE=ON \
            -DGINKGO_BUILD_TESTS=OFF \
            -DGINKGO_BUILD_BENCHMARKS=OFF \
            -DGINKGO_BUILD_EXAMPLES=OFF \
            -DGINKGO_BUILD_DOC=OFF \
            -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
            -DCMAKE_INSTALL_PREFIX="$GINKGO_DIR"
    log "Building Ginkgo (this can take 10-15 min)"
    CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
    LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        cmake --build "$GINKGO_SRC/build" --parallel "$(nproc)"
    cmake --install "$GINKGO_SRC/build"
    log "Ginkgo installed at $GINKGO_DIR"
}

case "$TARGET" in
    cutlass) setup_cutlass ;;
    ginkgo)  setup_ginkgo ;;
    all)     setup_cutlass; setup_ginkgo ;;
    *) echo "Usage: $0 [cutlass|ginkgo|all]" >&2; exit 1 ;;
esac

log "Done. Vendor libraries:"
log "  CUTLASS_DIR=$CUTLASS_DIR"
log "  GINKGO_DIR=$GINKGO_DIR"
