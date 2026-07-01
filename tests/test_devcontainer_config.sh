#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

test -f "$ROOT/.devcontainer/devcontainer.json"
test -f "$ROOT/.devcontainer/Dockerfile"

python3 - "$ROOT/.devcontainer/devcontainer.json" <<'PY'
import json
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1])
config = json.loads(config_path.read_text())

assert config["build"]["dockerfile"] == "Dockerfile"
assert config["containerEnv"]["OPENBLAS_ROOT"] == "/opt/openblas"
assert config["containerEnv"]["MKLROOT"] == "/opt/intel/oneapi/mkl/latest"
assert config["containerEnv"]["USE_CUDA"] == "0"
assert config["postCreateCommand"].startswith("uv sync")

mounts = "\n".join(config.get("mounts", []))
assert "/home/vscode/.cargo/registry" in mounts
assert "/home/vscode/.cache/uv" in mounts
PY

dockerfile="$(cat "$ROOT/.devcontainer/Dockerfile")"
case "$dockerfile" in
  *"ubuntu-24.04"* ) ;;
  * ) echo "Dockerfile must use an Ubuntu 24.04 base for Python 3.12" >&2; exit 1 ;;
esac

for required in \
    build-essential \
    cmake \
    intel-oneapi-mkl-devel \
    OPENBLAS_VERSION \
    OPENBLAS_MAX_THREADS \
    install_openblas.sh \
    python3.12-venv \
    ripgrep \
    rustup \
    uv; do
    grep -q "$required" "$ROOT/.devcontainer/Dockerfile"
done

if grep -q "libopenblas-dev" "$ROOT/.devcontainer/Dockerfile"; then
    echo "CPU devcontainer must build OpenBLAS instead of installing Ubuntu libopenblas-dev" >&2
    exit 1
fi

for required in \
    "USE_THREAD=1" \
    "USE_OPENMP=0" \
    "NUM_THREADS=" \
    "openblas_get_config" \
    "openblas_get_parallel" \
    "ldconfig"; do
    grep -q "$required" "$ROOT/.devcontainer/install_openblas.sh"
done

for required in \
    OPENBLAS_VERSION \
    OPENBLAS_MAX_THREADS \
    install_openblas.sh \
    "\${OPENBLAS_ROOT}/lib"; do
    grep -q "$required" "$ROOT/.devcontainer/cuda/Dockerfile"
done

if grep -q "libopenblas-dev" "$ROOT/.devcontainer/cuda/Dockerfile"; then
    echo "CUDA devcontainer must build OpenBLAS instead of installing Ubuntu libopenblas-dev" >&2
    exit 1
fi

grep -q 'devcontainer up --workspace-folder .' "$ROOT/README.md"
grep -q "devcontainer exec --workspace-folder ." "$ROOT/README.md"
grep -q 'BENCH_INSTANCE=bin_matmul_256' "$ROOT/README.md"
grep -q 'BENCHMARK_TARGET_PROFILE=amd-cpu' "$ROOT/README.md"
