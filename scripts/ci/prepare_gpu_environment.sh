#!/usr/bin/env bash
# Run only on the Ubuntu 24.04 x86_64 GitHub-hosted preparation job.
set -euo pipefail
runtime_root=/opt/tenferro-benchmark-ci
[[ "$(uname -m)" == x86_64 ]]
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]]
[[ ! -e "$runtime_root" ]]
started=$SECONDS
sudo mkdir -p "$runtime_root"
mkdir -p _gpu_environment
sudo chown "$(id -u):$(id -g)" "$runtime_root"
mkdir -p "$runtime_root/bin" "$runtime_root/cuda/lib64" "$runtime_root/cutensor/lib"

curl -fsSL --retry 3 --max-time 120 -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y --no-install-recommends zstd \
  cuda-cudart-12-8=12.8.90-1 cuda-nvrtc-12-8=12.8.93-1 \
  libcublas-12-8=12.8.5.5-1 libcusolver-12-8=11.7.3.90-1 \
  libcusparse-12-8=12.5.8.93-1 libnvjitlink-12-8=12.8.93-1 \
  cuda-cudart-dev-12-8=12.8.90-1 cuda-cccl-12-8=12.8.90-1 \
  cuda-crt-12-8=12.8.93-1 \
  libcutensor2=2.2.0.0-1
# Preserve soname symlinks, excluding compiler, static archives and driver stubs.
find /usr/local/cuda-12.8/targets/x86_64-linux/lib -maxdepth 1 -name '*.so*' \
  -exec cp -a -t "$runtime_root/cuda/lib64" {} +
find /usr/lib/x86_64-linux-gnu/libcutensor/12 -maxdepth 1 -name '*.so*' \
  -exec cp -a -t "$runtime_root/cutensor/lib" {} +
cp -a /usr/local/cuda-12.8/targets/x86_64-linux/include "$runtime_root/cuda/include"
dpkg-query -W 'cuda-cudart-12-8' 'cuda-nvrtc-12-8' 'libcublas-12-8' \
  'libcusolver-12-8' 'libcusparse-12-8' 'libnvjitlink-12-8' 'libcutensor2' \
  'cuda-cudart-dev-12-8' 'cuda-cccl-12-8' 'cuda-crt-12-8' > "$runtime_root/runtime-packages.txt"
python3 scripts/ci/check_cuda_headers.py --cuda-root "$runtime_root/cuda"

cp "$(command -v uv)" "$runtime_root/bin/uv"
export UV_PYTHON_INSTALL_DIR="$runtime_root/python"
export UV_PROJECT_ENVIRONMENT="$runtime_root/venv"
export UV_CACHE_DIR=/tmp/tenferro-gpu-preparation-cache
export UV_LINK_MODE=copy
uv python install 3.12.12 --no-bin
extra=()
if [[ ",${GPU_BENCH_BACKENDS:-}," == *,jax-cuda,* ]]; then
  extra=(--extra gpu)
fi
uv sync --python 3.12.12 --managed-python --frozen --no-dev "${extra[@]}"
"$runtime_root/venv/bin/python" - <<'PY'
import importlib.metadata as m
import json, pathlib, sys
assert sys.version_info[:3] == (3, 12, 12)
import torch, yaml, jsonschema, jax
pathlib.Path('/opt/tenferro-benchmark-ci/python-packages.json').write_text(
    json.dumps({d.metadata['Name']: d.version for d in m.distributions()}, sort_keys=True, indent=2) + '\n')
print('Prepared Python', sys.version, 'PyTorch', torch.__version__, 'JAX', jax.__version__)
PY
rm -rf "$UV_CACHE_DIR"
echo "Runtime preparation seconds: $((SECONDS-started))"
started=$SECONDS
tar -C /opt -cf - tenferro-benchmark-ci | zstd -T2 -3 -o _gpu_environment/runtime.tar.zst
zstd -dc _gpu_environment/runtime.tar.zst | python3 scripts/ci/gpu_environment.py check-tar
python3 scripts/ci/gpu_environment.py manifest --backends "${GPU_BENCH_BACKENDS:-}"
echo "Runtime compression seconds: $((SECONDS-started))"
du -h _gpu_environment/runtime.tar.zst
