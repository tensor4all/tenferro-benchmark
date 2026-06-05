#!/usr/bin/env bash

# Shared CPU BLAS provider policy for benchmark runners.
#
# macOS uses Accelerate for BLAS-backed tenferro CPU benchmarks so the Rust
# backend matches the BLAS provider used by the standard PyTorch wheel. JAX CPU
# dot is reported separately as an XLA CPU backend.
# Linux keeps the existing OpenBLAS default.

benchmark_host_os() {
    if [[ -n "${BENCHMARK_HOST_OS:-}" ]]; then
        printf '%s\n' "$BENCHMARK_HOST_OS"
    else
        uname -s
    fi
}

default_cpu_blas_features() {
    case "$(benchmark_host_os)" in
        Darwin) printf '%s\n' "system-accelerate" ;;
        *) printf '%s\n' "system-openblas" ;;
    esac
}

normalize_cpu_blas_features() {
    local features="${1:-}"
    if [[ -z "$features" ]]; then
        features="$(default_cpu_blas_features)"
    fi

    if [[ "$(benchmark_host_os)" == "Darwin" && "$features" == "system-openblas" ]]; then
        printf '%s\n' "system-accelerate"
        return
    fi

    printf '%s\n' "$features"
}

blas_impl_for_features() {
    case "$1" in
        system-openblas) printf '%s\n' "openblas" ;;
        system-accelerate) printf '%s\n' "accelerate" ;;
        *) printf '%s\n' "none" ;;
    esac
}

ensure_blas_env_for_features() {
    local features="$1"
    case "$features" in
        system-openblas)
            if [[ -n "${OPENBLAS_ROOT:-}" ]]; then
                export OPENBLAS_ROOT
                return
            fi
            cat >&2 <<'EOF'
OPENBLAS_ROOT is required for system-openblas.
Set OPENBLAS_ROOT to the OpenBLAS prefix, or choose system-accelerate on macOS.
EOF
            return 1
            ;;
        system-accelerate|cpu-faer|cuda)
            return
            ;;
        *)
            return
            ;;
    esac
}
