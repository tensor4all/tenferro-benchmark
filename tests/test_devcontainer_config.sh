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
assert config["containerEnv"]["PYTORCH_OPENBLAS_DIR"] == "${containerWorkspaceFolder}/extern/devcontainer/pytorch-openblas"
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
    libopenblas-dev \
    python3.12-venv \
    ripgrep \
    rustup \
    uv; do
    grep -q "$required" "$ROOT/.devcontainer/Dockerfile"
done

grep -q 'devcontainer up --workspace-folder .' "$ROOT/README.md"
grep -q "devcontainer exec --workspace-folder ." "$ROOT/README.md"
grep -q 'BENCH_INSTANCE=bin_matmul_256' "$ROOT/README.md"
grep -q 'PYTORCH_OPENBLAS_DIR' "$ROOT/README.md"
grep -q 'extern/devcontainer/pytorch-openblas' "$ROOT/README.md"
