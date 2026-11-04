#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin" "$TMP/project/.venv" "$TMP/wheels"
cat > "$TMP/project/uv.lock" <<'LOCK'
[[package]]
name = "torch"
version = "2.12.0"
source = { registry = "https://pypi.org/simple" }
LOCK
export UV_TEST_LOG="$TMP/uv.log"
cat > "$TMP/bin/uv" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$UV_TEST_LOG"
if [[ "${UV_TEST_FAIL:-}" == "${1:-}" ]]; then exit 1; fi
if [[ "${1:-}" == pip && "$*" == *'-r -'* ]]; then cat >/dev/null; fi
SH
cat > "$TMP/bin/uname" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "${TEST_OS:-Linux}"
SH
chmod +x "$TMP/bin/uv" "$TMP/bin/uname"
export PATH="$TMP/bin:$PATH"
unset BENCHMARK_TORCH_WHEEL BENCHMARK_TARGET_PROFILE UV_NO_SYNC UV_TEST_FAIL
source "$ROOT/scripts/python_venv.sh"

# The ordinary CPU baseline installs the lockfile without torch and then the
# CPU wheel at the pinned version, so the CUDA wheel is never downloaded (#342).
prepare_cpu_benchmark_python_venv "$TMP/project"
grep -qx 'venv --allow-existing .venv' "$UV_TEST_LOG"
grep -qx 'export --frozen --no-hashes --no-emit-project --prune torch' "$UV_TEST_LOG"
grep -qx 'pip install -r -' "$UV_TEST_LOG"
grep -qx 'pip install --index-url https://download.pytorch.org/whl/cpu --force-reinstall torch==2.12.0+cpu' "$UV_TEST_LOG"
! grep -q '^sync' "$UV_TEST_LOG"
test "$UV_NO_SYNC" = 1

# A lockfile without a readable torch pin must fail instead of guessing.
cp "$TMP/project/uv.lock" "$TMP/uv.lock.bak"
: > "$TMP/project/uv.lock"
: > "$UV_TEST_LOG"
unset UV_NO_SYNC
if prepare_cpu_benchmark_python_venv "$TMP/project"; then
    echo "missing locked torch version must fail" >&2
    exit 1
fi
mv "$TMP/uv.lock.bak" "$TMP/project/uv.lock"

# Source wheel survives a runner reset; resolve the image's symlink to its
# actual wheel filename (uv rejects a basename without PEP 427 wheel tags).
touch "$TMP/wheels/torch-2.12.0+openblas-cp312-cp312-linux_x86_64.whl"
ln -s "$TMP/wheels/torch-2.12.0+openblas-cp312-cp312-linux_x86_64.whl" "$TMP/wheels/torch.whl"
export BENCHMARK_TORCH_WHEEL="$TMP/wheels/torch.whl"
reset_benchmark_python_venv "$TMP/project"
test ! -e "$TMP/project/.venv"
: > "$UV_TEST_LOG"
prepare_cpu_benchmark_python_venv "$TMP/project"
grep -qx 'venv --allow-existing .venv' "$UV_TEST_LOG"
grep -qx 'export --frozen --no-hashes --no-emit-project --prune torch' "$UV_TEST_LOG"
grep -Fxq "pip install --reinstall-package torch -r - $(realpath "$BENCHMARK_TORCH_WHEEL")" "$UV_TEST_LOG"
! grep -q 'download.pytorch.org' "$UV_TEST_LOG"
test "$UV_NO_SYNC" = 1

# Setup failure must propagate, even when the function is called in an if/&&.
for UV_TEST_FAIL in venv export pip; do
    export UV_TEST_FAIL
    unset UV_NO_SYNC
    if prepare_cpu_benchmark_python_venv "$TMP/project"; then
        echo "setup unexpectedly succeeded after $UV_TEST_FAIL failure" >&2
        exit 1
    fi
    test -z "${UV_NO_SYNC:-}"
done
unset UV_TEST_FAIL
export BENCHMARK_TORCH_WHEEL="$TMP/missing.whl"
: > "$UV_TEST_LOG"
if prepare_cpu_benchmark_python_venv "$TMP/project"; then
    echo "missing source wheel must not fall back to MKL" >&2
    exit 1
fi
test ! -s "$UV_TEST_LOG"

# macOS CPU benchmarks also reset the venv, so recreate it from the lockfile.
unset BENCHMARK_TORCH_WHEEL
: > "$UV_TEST_LOG"
TEST_OS=Darwin prepare_cpu_benchmark_python_venv "$TMP/project"
grep -qx 'sync --frozen' "$UV_TEST_LOG"
test "$UV_NO_SYNC" = 1

# GPU setup remains unchanged.
: > "$UV_TEST_LOG"
BENCHMARK_TARGET_PROFILE=nvidia-gpu prepare_cpu_benchmark_python_venv "$TMP/project"
test ! -s "$UV_TEST_LOG"
echo 'Python venv provider selection: PASS'
