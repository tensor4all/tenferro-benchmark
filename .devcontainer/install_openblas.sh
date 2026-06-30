#!/usr/bin/env bash
set -euo pipefail

: "${OPENBLAS_ROOT:=/opt/openblas}"
: "${OPENBLAS_VERSION:=0.3.26}"
: "${OPENBLAS_MAX_THREADS:=64}"
: "${OPENBLAS_DYNAMIC_ARCH:=1}"

build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

curl -fsSL "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz" \
    | tar -xz -C "$build_dir" --strip-components=1

make_flags=(
    "USE_THREAD=1"
    "USE_OPENMP=0"
    "NUM_THREADS=${OPENBLAS_MAX_THREADS}"
    "NO_AFFINITY=1"
)

if [[ "$OPENBLAS_DYNAMIC_ARCH" == "1" ]]; then
    make_flags+=("DYNAMIC_ARCH=1")
fi

make -C "$build_dir" -j"$(nproc)" "${make_flags[@]}"
make -C "$build_dir" install PREFIX="$OPENBLAS_ROOT"

echo "$OPENBLAS_ROOT/lib" > /etc/ld.so.conf.d/openblas.conf
ldconfig

OPENBLAS_LIB="$OPENBLAS_ROOT/lib/libopenblas.so" python3 <<'PY'
import ctypes
import os
import sys

lib = ctypes.CDLL(os.environ["OPENBLAS_LIB"])
lib.openblas_get_config.restype = ctypes.c_char_p
lib.openblas_get_parallel.restype = ctypes.c_int

config = lib.openblas_get_config().decode()
parallel = lib.openblas_get_parallel()
print(config)
if "SINGLE_THREADED" in config or parallel != 1:
    print(
        f"OpenBLAS build is not pthread-parallel: config={config!r}, parallel={parallel}",
        file=sys.stderr,
    )
    sys.exit(1)
PY
