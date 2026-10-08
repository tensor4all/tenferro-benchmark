#!/usr/bin/env bash
# Restore at the identical absolute prefix used by the hosted preparation job.
set -euo pipefail
started=$SECONDS
runtime_root=/opt/tenferro-benchmark-ci
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]]
[[ ! -e "$runtime_root" ]]
python3 scripts/ci/gpu_environment.py assemble
python3 scripts/ci/gpu_environment.py verify --backends "$GPU_BENCH_BACKENDS" --sha256 "$GPU_RUNTIME_SHA256"
zstd -dc _gpu_environment/runtime.tar.zst | python3 scripts/ci/gpu_environment.py check-tar
zstd -dc _gpu_environment/runtime.tar.zst | tar -xf - -C /opt
export UV_PROJECT_ENVIRONMENT="$runtime_root/venv"
export UV_PYTHON_INSTALL_DIR="$runtime_root/python"
export UV_NO_SYNC=1 UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never
export PATH="$runtime_root/bin:$runtime_root/venv/bin:$PATH"
export CUDA_HOME="$runtime_root/cuda" CUDA_PATH="$runtime_root/cuda"
export TENFERRO_CUTENSOR_PATH="$runtime_root/cutensor/lib/libcutensor.so.2"
export LD_LIBRARY_PATH="$runtime_root/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uv run python - <<'PY'
import sys, torch, jax, yaml, jsonschema
assert sys.version_info[:3] == (3, 12, 12)
print('Restored Python', sys.version, 'PyTorch', torch.__version__, 'JAX', jax.__version__)
PY
for name in UV_PROJECT_ENVIRONMENT UV_PYTHON_INSTALL_DIR UV_NO_SYNC UV_OFFLINE UV_PYTHON_DOWNLOADS CUDA_HOME CUDA_PATH TENFERRO_CUTENSOR_PATH LD_LIBRARY_PATH; do
  printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
done
printf '%s\n' "$runtime_root/bin" "$runtime_root/venv/bin" >> "$GITHUB_PATH"
echo "Runtime verification/extraction seconds: $((SECONDS-started))"
