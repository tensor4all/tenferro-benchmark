#!/usr/bin/env bash
# Per-arm launch policy shared by scripts/run_gpu_suite.sh and scripts/run_gpu_public_api.sh (#335).
#
#   gpu_arm_prepare <run_dir> <backend> [<output>]
#       Decide how one arm starts and write <run_dir>/arms/<backend>.json before it does.
#       Sets the array GPU_ARM_PREFIX (empty, or `taskset -c <cpu>`), and returns
#         0   run the arm with "${GPU_ARM_PREFIX[@]}" in front of its command;
#         10  skip the arm: <output> already holds rows from an earlier process, so the
#             arm is not started and its record says so instead of describing a run whose
#             output it did not write;
#         76  no CPU of the allocation is quiet enough (scripts/gpu_measurement_cpu.py names
#             the cpuset and every CPU's busy fraction); release the allocation and retry.
#   gpu_arm_finish <run_dir> <backend> <exit_code>
#       Add the arm's exit code and finish time to its record.
#
# GPU_BENCH_CPU_SELECT=1 turns on the CPU choice: the quietest CPU of the process
# affinity (or GPU_BENCH_CPUSET), under GPU_BENCH_CPU_MAX_BUSY (default 0.05). Unset, the
# arms run unpinned, as before, and the record says the selection was disabled.

GPU_ARM_SKIPPED=10
GPU_ARM_NO_USABLE_CPU=76

_gpu_arm_script_dir() { cd "$(dirname "${BASH_SOURCE[0]}")" && pwd; }

gpu_arm_prepare() {
    local run_dir="$1" backend="$2" output="${3:-}"
    local record="$run_dir/arms/$backend.json" cpu rc
    GPU_ARM_PREFIX=()
    mkdir -p "$run_dir/arms"
    if [[ -n "$output" && -s "$output" ]]; then
        python3 - "$record" "$backend" "$output" <<'PY'
import json, sys, time
record, backend, output = sys.argv[1:]
rows = sum(1 for line in open(output) if line.strip())
json.dump({"arm": backend, "status": "skipped",
           "reason": f"{output} already holds {rows} rows written by an earlier process; "
                     "the arm was not started, so this record describes no run",
           "output": output, "rows": rows,
           "skipped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
          open(record, "w"), indent=2)
PY
        [[ $? == 0 ]] || return 1
        echo "SKIP $backend: $output already holds rows from an earlier process (recorded in $record)" >&2
        return "$GPU_ARM_SKIPPED"
    fi
    if [[ "${GPU_BENCH_CPU_SELECT:-0}" != 1 ]]; then
        python3 - "$record" "$backend" <<'PY'
import json, sys
record, backend = sys.argv[1:]
json.dump({"arm": backend, "status": "disabled",
           "reason": "GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned"},
          open(record, "w"), indent=2)
PY
        [[ $? == 0 ]] || return 1
        return 0
    fi
    if ! command -v taskset >/dev/null 2>&1; then
        echo "ERROR: GPU_BENCH_CPU_SELECT=1 needs taskset (util-linux)" >&2
        return 1
    fi
    rc=0
    cpu="$(python3 "$(_gpu_arm_script_dir)/gpu_measurement_cpu.py" select \
        --arm "$backend" --record "$record")" || rc=$?
    if [[ "$rc" != 0 ]]; then
        return "$rc"
    fi
    GPU_ARM_PREFIX=(taskset -c "$cpu")
    echo "  $backend: measurement CPU $cpu (record $record)"
}

gpu_arm_finish() {
    local run_dir="$1" backend="$2" exit_code="$3"
    local record="$run_dir/arms/$backend.json"
    [[ -f "$record" ]] || return 0
    python3 - "$record" "$exit_code" <<'PY'
import json, sys, time
path, code = sys.argv[1], int(sys.argv[2])
record = json.load(open(path))
record.update(exit_code=code, finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
json.dump(record, open(path, "w"), indent=2)
PY
}

# Exit the calling driver on a prepare result that is neither run nor skip.
gpu_arm_fail_unless_runnable() {
    local rc="$1"
    if [[ "$rc" == "$GPU_ARM_NO_USABLE_CPU" ]]; then
        echo "Stopping: no usable measurement CPU in this allocation (exit $GPU_ARM_NO_USABLE_CPU)." >&2
    fi
    exit "$rc"
}
