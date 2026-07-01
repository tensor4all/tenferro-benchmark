#!/usr/bin/env bash

# Shared CPU BLAS provider policy for benchmark runners.
#
# macOS uses Accelerate for BLAS-backed tenferro CPU benchmarks so the Rust
# backend matches the BLAS provider used by the standard PyTorch wheel. JAX CPU
# dot is reported separately as an XLA CPU backend.
# Linux keeps the existing OpenBLAS default. system-mkl is opt-in.

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
        system-mkl) printf '%s\n' "mkl" ;;
        *) printf '%s\n' "none" ;;
    esac
}

prepend_ld_library_path() {
    local dir="$1"
    [[ -d "$dir" ]] || return 0
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$dir:"*) ;;
        *) export LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
}

ensure_blas_env_for_features() {
    local features="$1"
    case "$features" in
        system-openblas)
            if [[ -n "${OPENBLAS_ROOT:-}" ]]; then
                export OPENBLAS_ROOT
                prepend_ld_library_path "$OPENBLAS_ROOT/lib"
                return
            fi
            cat >&2 <<'EOF'
OPENBLAS_ROOT is required for system-openblas.
Set OPENBLAS_ROOT to the OpenBLAS prefix, or choose system-accelerate on macOS.
EOF
            return 1
            ;;
        system-mkl)
            if [[ -z "${MKLROOT:-}" ]]; then
                cat >&2 <<'EOF'
MKLROOT is required for system-mkl.
Set MKLROOT to the Intel oneMKL prefix, for example /opt/intel/oneapi/mkl/latest.
EOF
                return 1
            fi
            if [[ ! -d "$MKLROOT/lib" && ! -d "$MKLROOT/lib/intel64" ]]; then
                echo "MKLROOT lib directory does not exist: $MKLROOT/lib" >&2
                return 1
            fi
            if [[ ! -d "$MKLROOT/include" ]]; then
                echo "MKLROOT include directory does not exist: $MKLROOT/include" >&2
                return 1
            fi
            export MKLROOT
            prepend_ld_library_path "$MKLROOT/lib"
            prepend_ld_library_path "$MKLROOT/lib/intel64"
            prepend_ld_library_path "/opt/intel/oneapi/compiler/latest/lib"
            return
            ;;
        system-accelerate|cpu-faer|cuda)
            return
            ;;
        *)
            return
            ;;
    esac
}
