#!/usr/bin/env bash
set -euo pipefail
export RUNNER_ALLOW_RUNASROOT=1
cd /home/runner
./bin/Runner.Listener --version
./externals/node24/bin/node --version
command -v git jq zstd nm python3 cargo cargo-nextest
cargo --version
cargo nextest --version
for tier in 12.6 12.8; do
  python3 /source/scripts/ci/check_cuda_headers.py --cuda-root "/usr/local/cuda-$tier"
  LD_LIBRARY_PATH="/usr/local/cuda-$tier/lib64" python3 - "$tier" <<'PY'
import ctypes,sys
from pathlib import Path
root=Path('/usr/local/cuda-'+sys.argv[1])
lib=ctypes.CDLL(str(root/'lib64/libnvrtc.so.12'))
a,b=ctypes.c_int(),ctypes.c_int()
assert lib.nvrtcVersion(ctypes.byref(a),ctypes.byref(b))==0
assert f'{a.value}.{b.value}'==sys.argv[1]
ctypes.CDLL('/opt/tenferro-ci/cutensor-2.6.0.4/lib/libcutensor.so.2')
print('Verified loaded NVRTC and cuTENSOR',sys.argv[1])
PY
done
python3 - <<'PY'
import ctypes,json
from pathlib import Path
root=Path('/opt/ci-cost-runtime')
manifest=json.loads((root/'prepared-image.json').read_text())
assert manifest['cuda_tiers']==['12.6','12.8']
for pattern in ['*/jax_plugins/xla_cuda12/xla_cuda_plugin.so','*/nvidia/cudnn/lib/libcudnn.so.9',
                '*/nvidia/cuda_nvcc/bin/ptxas','*/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc']:
 matches=list((root/'wheels-unpacked').glob(pattern))
 assert len(matches)==1, (pattern,matches)
 print('Verified prepared dependency', matches[0])
PY
plugin=$(find /opt/ci-cost-runtime/wheels-unpacked -name xla_cuda_plugin.so -print -quit)
nm -D "$plugin" > /tmp/plugin-symbols
# Avoid a SIGPIPE-dependent pipeline exit under pipefail.
grep -q GetPjrtApi /tmp/plugin-symbols
ptxas=$(find /opt/ci-cost-runtime/wheels-unpacked -path '*/bin/ptxas' -print -quit)
"$ptxas" --version
printf 'PREPARED_IMAGE_NATIVE_VALIDATION_PASSED\n'
