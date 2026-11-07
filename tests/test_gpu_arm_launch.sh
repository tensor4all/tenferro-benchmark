#!/usr/bin/env bash
# #335: per-arm launch policy of the GPU drivers (scripts/gpu_arm_launch.sh), no GPU needed.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gpu_arm_launch.sh
source scripts/gpu_arm_launch.sh
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }
field() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }

# 1. An output an earlier process wrote is skipped explicitly, not run and recorded as failed.
printf '{"a":1}\n{"a":2}\n' > "$TMP/direct.jsonl"
rc=0; gpu_arm_prepare "$TMP/run" tenferro-cuda-direct "$TMP/direct.jsonl" 2>/dev/null || rc=$?
[[ "$rc" == "$GPU_ARM_SKIPPED" ]] || fail "existing output: rc=$rc"
[[ "$(field "$TMP/run/arms/tenferro-cuda-direct.json" status)" == skipped ]] || fail "skip record"
[[ "$(field "$TMP/run/arms/tenferro-cuda-direct.json" rows)" == 2 ]] || fail "skip rows"
[[ "$(wc -l < "$TMP/direct.jsonl")" == 2 ]] || fail "existing output was touched"

# 2. Selection off: the arm runs unpinned and the record says so.
unset GPU_BENCH_CPU_SELECT
rc=0; gpu_arm_prepare "$TMP/run" pytorch-cuda "$TMP/pytorch.jsonl" || rc=$?
[[ "$rc" == 0 && ${#GPU_ARM_PREFIX[@]} == 0 ]] || fail "disabled: rc=$rc prefix=${GPU_ARM_PREFIX[*]}"
[[ "$(field "$TMP/run/arms/pytorch-cuda.json" status)" == disabled ]] || fail "disabled record"
gpu_arm_finish "$TMP/run" pytorch-cuda 0
[[ "$(field "$TMP/run/arms/pytorch-cuda.json" exit_code)" == 0 ]] || fail "finish exit code"

# 3. Selection on: the quietest CPU of the allocation is taken and pinned.
export GPU_BENCH_CPU_SELECT=1 GPU_BENCH_CPU_IDLE_WINDOW_SECONDS=0 GPU_BENCH_CPUSET=0-1
export BENCHMARK_PROC_STAT="$TMP/stat"
printf 'cpu0 0 0 0 0 0 0 0 0 0 0\ncpu1 0 0 0 0 0 0 0 0 0 0\n' > "$BENCHMARK_PROC_STAT"
# /proc/stat does not move in a zero window, so every CPU reads 0 busy: the lowest id wins.
rc=0; gpu_arm_prepare "$TMP/run" jax-cuda >/dev/null || rc=$?
[[ "$rc" == 0 ]] || fail "select: rc=$rc"
[[ "${GPU_ARM_PREFIX[*]}" == "taskset -c 0" ]] || fail "select prefix: ${GPU_ARM_PREFIX[*]}"
[[ "$(field "$TMP/run/arms/jax-cuda.json" cpu)" == 0 ]] || fail "select record"

# 4. A busy allocation fails with 76 and a message naming the cpuset.
cat > "$TMP/busy_stat.py" <<'PY'
import sys, pathlib
path = pathlib.Path(sys.argv[1])
n = int(path.read_text().split()[1]) + 50 if path.exists() else 0
path.write_text(f"cpu0 {n} 0 0 {n} 0 0 0 0 0 0\ncpu1 {n} 0 0 {n} 0 0 0 0 0 0\n")
PY
# Each read of the fake stat file advances it by 50 busy + 50 idle jiffies, i.e. 50% busy.
python3 "$TMP/busy_stat.py" "$BENCHMARK_PROC_STAT"
python3 - "$TMP" <<'PY'
import pathlib, sys
tmp = pathlib.Path(sys.argv[1])
# A sitecustomize that advances the fake stat around time.sleep, so the select window sees load.
(tmp / "sitecustomize.py").write_text(
    "import time, subprocess, sys, os\n"
    "_s = time.sleep\n"
    "def sleep(x):\n"
    "    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'busy_stat.py'), os.environ['BENCHMARK_PROC_STAT']], check=True)\n"
    "time.sleep = sleep\n")
PY
rc=0; PYTHONPATH="$TMP" gpu_arm_prepare "$TMP/run" jax-cuda-jit 2> "$TMP/err" >/dev/null || rc=$?
[[ "$rc" == "$GPU_ARM_NO_USABLE_CPU" ]] || fail "busy: rc=$rc ($(cat "$TMP/err"))"
grep -q "no usable measurement CPU in cpuset 0-1" "$TMP/err" || fail "busy message: $(cat "$TMP/err")"
grep -q "cpu0=0.500" "$TMP/err" || fail "busy per-CPU listing"
[[ "$(field "$TMP/run/arms/jax-cuda-jit.json" status)" == no_usable_cpu ]] || fail "busy record"

echo "gpu arm launch policy test passed"
