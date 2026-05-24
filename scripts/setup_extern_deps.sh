#!/usr/bin/env bash
set -euo pipefail

# Prepare repo-local external dependencies used by the benchmark suite.
# This script is intended to be sourced by run_all.sh so exported variables
# remain available to child benchmark scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXTERN_DIR="$PROJECT_DIR/extern"

TENFERRO_REPO_URL="${TENFERRO_REPO_URL:-https://github.com/tensor4all/tenferro-rs.git}"
TENFERRO_REF="${TENFERRO_REF:-}"
PYTORCH_REPO_URL="${PYTORCH_REPO_URL:-https://github.com/pytorch/pytorch.git}"
PYTORCH_REF="${PYTORCH_REF:-v2.12.0}"

TENFERRO_DIR="${TENFERRO_RS_DIR:-$EXTERN_DIR/tenferro-rs}"
PYTORCH_DIR="${PYTORCH_OPENBLAS_DIR:-$EXTERN_DIR/pytorch-openblas}"
PYTORCH_VENV="$PYTORCH_DIR/.venv-openblas"

log() {
    printf '[setup_extern_deps] %s\n' "$*" >&2
}

ensure_openblas_root() {
    if [[ -n "${OPENBLAS_ROOT:-}" ]]; then
        export OPENBLAS_ROOT
        return
    fi
    if command -v brew >/dev/null 2>&1 && brew --prefix openblas >/dev/null 2>&1; then
        export OPENBLAS_ROOT="$(brew --prefix openblas)"
        return
    fi
    cat >&2 <<'EOF'
OPENBLAS_ROOT is required.
Set OPENBLAS_ROOT to the OpenBLAS prefix, or install OpenBLAS with Homebrew.
EOF
    exit 1
}

clone_or_reuse_checkout() {
    local name="$1"
    local url="$2"
    local ref="$3"
    local dest="$4"
    local sibling="$PROJECT_DIR/../$name"

    mkdir -p "$EXTERN_DIR"
    if [[ -d "$dest/.git" ]]; then
        log "$name already exists at ${dest#$PROJECT_DIR/}"
        return
    fi
    if [[ -e "$dest" ]]; then
        log "$name exists at ${dest#$PROJECT_DIR/}; assuming it is usable"
        return
    fi

    if [[ "${SETUP_EXTERN_MIGRATE_SIBLINGS:-1}" == "1" && -d "$sibling/.git" ]]; then
        log "moving existing ../$name into extern/$name"
        mv "$sibling" "$dest"
        return
    fi

    log "cloning $name into ${dest#$PROJECT_DIR/}"
    if [[ -n "$ref" ]]; then
        git clone --recursive --branch "$ref" --depth 1 --shallow-submodules "$url" "$dest"
    else
        git clone --recursive "$url" "$dest"
    fi
}

ensure_tenferro_checkout() {
    clone_or_reuse_checkout "tenferro-rs" "$TENFERRO_REPO_URL" "$TENFERRO_REF" "$TENFERRO_DIR"
}

ensure_pytorch_checkout() {
    clone_or_reuse_checkout "pytorch-openblas" "$PYTORCH_REPO_URL" "$PYTORCH_REF" "$PYTORCH_DIR"
    if [[ ! -d "$PYTORCH_DIR/third_party/protobuf" ]]; then
        log "initializing PyTorch submodules"
        git -C "$PYTORCH_DIR" submodule update --init --recursive --depth 1
    fi
}

libtorch_cpu_path() {
    find "$PYTORCH_DIR/torch/lib" -name 'libtorch_cpu.*' -print 2>/dev/null | head -n 1 || true
}

libtorch_links_openblas() {
    local libtorch_cpu="$1"
    [[ -n "$libtorch_cpu" ]] || return 1
    if command -v otool >/dev/null 2>&1; then
        otool -L "$libtorch_cpu" | grep -qi openblas
    elif command -v ldd >/dev/null 2>&1; then
        ldd "$libtorch_cpu" | grep -qi openblas
    else
        return 1
    fi
}

ensure_pytorch_openblas_build() {
    local torch_config="$PYTORCH_DIR/torch/share/cmake/Torch/TorchConfig.cmake"
    local libtorch_cpu
    libtorch_cpu="$(libtorch_cpu_path)"
    if [[ -f "$torch_config" ]] && libtorch_links_openblas "$libtorch_cpu"; then
        log "OpenBLAS-linked PyTorch is ready at extern/pytorch-openblas"
        return
    fi

    log "building PyTorch $PYTORCH_REF with OpenBLAS; this can take a long time"
    "${PYTHON_BIN:-python3}" -m venv "$PYTORCH_VENV"
    "$PYTORCH_VENV/bin/python" -m pip install --upgrade pip setuptools wheel
    (cd "$PYTORCH_DIR" && "$PYTORCH_VENV/bin/python" -m pip install --group dev)

    (
        cd "$PYTORCH_DIR"
        export BLAS=OpenBLAS
        export OpenBLAS_HOME="$OPENBLAS_ROOT"
        export CMAKE_PREFIX_PATH="$OPENBLAS_ROOT:${CMAKE_PREFIX_PATH:-}"
        export CMAKE_POLICY_VERSION_MINIMUM="${CMAKE_POLICY_VERSION_MINIMUM:-3.5}"
        export USE_CUDA="${USE_CUDA:-0}"
        export USE_ROCM="${USE_ROCM:-0}"
        export USE_MPS="${USE_MPS:-0}"
        export USE_MKLDNN="${USE_MKLDNN:-0}"
        export USE_DISTRIBUTED="${USE_DISTRIBUTED:-0}"
        export USE_NNPACK="${USE_NNPACK:-0}"
        export USE_QNNPACK="${USE_QNNPACK:-0}"
        export USE_PYTORCH_QNNPACK="${USE_PYTORCH_QNNPACK:-0}"
        export USE_XNNPACK="${USE_XNNPACK:-0}"
        export USE_FBGEMM="${USE_FBGEMM:-0}"
        export BUILD_TEST="${BUILD_TEST:-0}"
        export MAX_JOBS="${MAX_JOBS:-8}"
        "$PYTORCH_VENV/bin/python" -m pip install --no-build-isolation -v -e .
    )

    libtorch_cpu="$(libtorch_cpu_path)"
    if ! libtorch_links_openblas "$libtorch_cpu"; then
        echo "PyTorch build completed, but libtorch_cpu does not link OpenBLAS: $libtorch_cpu" >&2
        exit 1
    fi
}

main() {
    ensure_openblas_root
    ensure_tenferro_checkout
    ensure_pytorch_checkout
    ensure_pytorch_openblas_build

    export TENFERRO_RS_DIR="$TENFERRO_DIR"
    export PYTORCH_OPENBLAS_DIR="$PYTORCH_DIR"
    export Torch_DIR="$PYTORCH_DIR/torch/share/cmake/Torch"

    log "OPENBLAS_ROOT=$OPENBLAS_ROOT"
    log "TENFERRO_RS_DIR=$TENFERRO_RS_DIR"
    log "Torch_DIR=$Torch_DIR"
}

main "$@"
