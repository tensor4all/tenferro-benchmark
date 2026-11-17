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
# tenferro-rs 0.12 links no vendor BLAS/LAPACK provider (tenferro-rs #2608), so
# the image no longer sets a provider prefix for the harness build.
assert "OPENBLAS_ROOT" not in config["containerEnv"]
assert "MKLROOT" not in config["containerEnv"]
assert config["containerEnv"]["USE_CUDA"] == "0"
command = config["postCreateCommand"]
assert "prepare_cpu_benchmark_python_venv" in command
assert "uv sync" not in command

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
    python3.12-venv \
    ripgrep \
    rustup \
    uv; do
    grep -q "$required" "$ROOT/.devcontainer/Dockerfile"
done

# The provider-selectable images are gone with the vendor backend: no image may
# keep an OpenBLAS/MKL prefix or a provider source build for the harness.
for image in .devcontainer/Dockerfile .devcontainer/cuda/Dockerfile; do
    if grep -q "OPENBLAS_ROOT\|install_openblas.sh\|intel-oneapi-mkl" "$ROOT/$image"; then
        echo "$image must not install or configure a vendor BLAS/LAPACK provider" >&2
        exit 1
    fi
done
test ! -e "$ROOT/.devcontainer/install_openblas.sh"
test ! -e "$ROOT/.devcontainer/openblas"

python3 - "$ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
config = json.loads((root / ".devcontainer/cuda/devcontainer.json").read_text())
assert config["initializeCommand"] == ["mkdir", "-p", "${localWorkspaceFolder}/.venv"]
assert config["containerEnv"]["UV_NO_SYNC"] == "1"
assert "OPENBLAS_ROOT" not in config["containerEnv"]
mounts = "\n".join(config.get("mounts", []))
assert "source=tenferro-benchmark-cuda-venv-${devcontainerId}" in mounts
assert "target=${containerWorkspaceFolder}/.venv,type=volume" in mounts
command = config["postCreateCommand"]
assert command.startswith("sudo mkdir -p")
assert "chown -R vscode:vscode" in command
PY

grep -q 'devcontainer up --workspace-folder .' "$ROOT/README.md"
grep -q "devcontainer exec --workspace-folder ." "$ROOT/README.md"
grep -q 'BENCH_INSTANCE=bin_matmul_256' "$ROOT/README.md"
grep -q 'BENCHMARK_TARGET_PROFILE=amd-cpu' "$ROOT/README.md"
