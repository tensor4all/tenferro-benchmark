#!/usr/bin/env bash
# Run inside the declared disposable image, with the frozen runtime mounted.
set -euo pipefail
test -f /.dockerenv
test "$(id -u)" -eq 0
arm="$1"
case "$arm" in baseline|candidate) ;; *) exit 2 ;; esac
. /etc/os-release
test "$ID" = ubuntu
test "$VERSION_ID" = 24.04
export DEBIAN_FRONTEND=noninteractive
export RUNNER_ALLOW_RUNASROOT=1
apt-get update
if [ "$arm" = baseline ]; then
  apt-get install -y curl tar gzip ca-certificates python3 python3-pip
else
  command -v curl tar gzip python3
fi
# Exercise the real declared installer without the GPU-only launch proof.
python3 - "$arm" <<'PY'
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('smoke', f'/validation/{sys.argv[1]}-smoke.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.install_nvrtc((12, 8))
PY
test "$(dpkg-query -W -f='${Version}' cuda-nvrtc-12-8)" = 12.8.93-1
if [ "$arm" = baseline ]; then
  mkdir /tmp/native-runner
  cd /tmp/native-runner
  curl -fsSL --retry 2 -o runner.tar.gz \
    https://github.com/actions/runner/releases/download/v2.337.0/actions-runner-linux-x64-2.337.0.tar.gz
  echo '70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613  runner.tar.gz' | sha256sum -c -
  tar -xzf runner.tar.gz
  bash bin/installdependencies.sh
  ./bin/Runner.Listener --version
  apt-get install -y git jq zstd binutils libssl3t64
else
  /home/runner/bin/Runner.Listener --version
  apt-get install -y --no-install-recommends git jq zstd binutils libssl3t64
fi
export PATH="/runtime/common/bin:$PATH"
export LD_LIBRARY_PATH="/runtime/sdk/lib64:/runtime/common/opt/tenferro-ci/cutensor-2.6.0.4/lib:${LD_LIBRARY_PATH:-}"
cargo --version
cargo nextest --version
python3 /validation/check_cuda_headers.py --cuda-root /runtime/sdk
python3 - <<'PY'
import ctypes
from pathlib import Path
ctypes.CDLL('/runtime/common/opt/tenferro-ci/cutensor-2.6.0.4/lib/libcutensor.so.2')
root=Path('/runtime/common/wheels-unpacked')
assert len(list(root.glob('jax_cuda12_pjrt-*/jax_plugins/xla_cuda12/xla_cuda_plugin.so')))==1
assert list(root.glob('nvidia_cudnn_cu12-*/nvidia/cudnn/lib/libcudnn.so*'))
assert list(root.glob('nvidia_cuda_nvcc_cu12-*/nvidia/cuda_nvcc/bin/ptxas'))
print('cuTENSOR load and complete prepared PJRT components verified')
PY
for executable in /runtime/common/wheels-unpacked/nvidia_cuda_nvcc_cu12-*/nvidia/cuda_nvcc/bin/ptxas; do
  "$executable" --version
done
echo "NATIVE_PREFLIGHT_PASSED=$arm"
