#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin" "$TMP/project/.venv" "$TMP/wheels"
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

# Preserve the ordinary CPU wheel baseline.
prepare_cpu_benchmark_python_venv "$TMP/project"
grep -qx 'sync' "$UV_TEST_LOG"
grep -qx 'pip install --index-url https://download.pytorch.org/whl/cpu --force-reinstall torch==2.12.0+cpu' "$UV_TEST_LOG"
test "$UV_NO_SYNC" = 1

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

# The new opt-in must not change macOS or GPU setup.
TEST_OS=Darwin prepare_cpu_benchmark_python_venv "$TMP/project"
BENCHMARK_TARGET_PROFILE=nvidia-gpu prepare_cpu_benchmark_python_venv "$TMP/project"
test ! -s "$UV_TEST_LOG"
echo 'Python venv provider selection: PASS'
