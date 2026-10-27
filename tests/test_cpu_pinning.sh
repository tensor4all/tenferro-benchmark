#!/usr/bin/env bash
# zen5-cpu pins each thread row under taskset and records the masks; other
# profiles are left unpinned. Needs no timing and no benchmark build.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/cpu_pinning.sh"

fail() { echo "FAIL: $*" >&2; exit 1; }
expect() { [[ "$1" == "$2" ]] || fail "$3: expected '$2', got '$1'"; }

expect "$(cpu_pinning_measured_mask zen5-cpu 1)" "0" "1T measured"
expect "$(cpu_pinning_measured_mask zen5-cpu 4)" "0-3" "4T measured"
expect "$(cpu_pinning_measured_mask zen5-cpu 8)" "4-11" "8T measured"
expect "$(cpu_pinning_observer_mask zen5-cpu 1)" "4-11,16-23" "1T observer"
expect "$(cpu_pinning_observer_mask zen5-cpu 4)" "4-11,16-23" "4T observer"
expect "$(cpu_pinning_observer_mask zen5-cpu 8)" "0-3,12-15" "8T observer"
expect "$(cpu_pinning_measured_mask amd-cpu 4)" "" "amd-cpu is unpinned"
expect "$(cpu_pinning_measured_mask mac-cpu 1)" "" "mac-cpu is unpinned"
status=0
cpu_pinning_measured_mask zen5-cpu 2 >/dev/null || status=$?
expect "$status" "2" "unknown zen5-cpu row"

# The mask set for a row is disjoint from its observer and never shares an SMT sibling.
disjoint() {
    python3 - "$1" "$2" <<'PY'
import sys
def cpus(text):
    out = set()
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out
measured, observer = (cpus(arg) for arg in sys.argv[1:3])
siblings = {c + 12 if c < 12 else c - 12 for c in measured}  # SMT2: cpu N pairs with N+12
assert not (measured & observer), "measured and observer overlap"
assert not (siblings & observer), "observer sits on an SMT sibling of a measured core"
PY
}
for threads in 1 4 8; do
    disjoint "$(cpu_pinning_measured_mask zen5-cpu "$threads")" "$(cpu_pinning_observer_mask zen5-cpu "$threads")" \
        || fail "$threads-thread masks overlap"
done

if command -v taskset >/dev/null 2>&1 && [[ "$(nproc --all)" -ge 12 ]]; then
    (
        export BENCHMARK_TARGET_PROFILE=zen5-cpu
        source "$ROOT_DIR/scripts/thread_env.sh"
        for threads in 1 4 8; do
            configure_cpu_thread_env "$threads"
            want="$(cpu_pinning_measured_mask zen5-cpu "$threads")"
            got="$(taskset -pc $$ | sed 's/.*: //')"
            expect "$got" "$want" "$threads-thread shell affinity"
            expect "$BENCHMARK_MEASURED_CPUS" "$want" "$threads-thread recorded mask"
            expect "$RAYON_NUM_THREADS" "$threads" "$threads-thread budget"
        done
        if configure_cpu_thread_env 2 2>/dev/null; then fail "2 threads must be refused on zen5-cpu"; fi
    )
    (
        export BENCHMARK_TARGET_PROFILE=amd-cpu
        source "$ROOT_DIR/scripts/thread_env.sh"
        before="$(taskset -pc $$ | sed 's/.*: //')"
        configure_cpu_thread_env 4
        expect "$(taskset -pc $$ | sed 's/.*: //')" "$before" "amd-cpu affinity must not change"
        expect "${BENCHMARK_MEASURED_CPUS:-}" "" "amd-cpu records no pinning"
    )
fi

# Run metadata records what the measured process actually had.
OUT="$(cd "$ROOT_DIR/scripts" && BENCHMARK_CPU_PINNING=taskset BENCHMARK_MEASURED_CPUS=4-11 \
    BENCHMARK_OBSERVER_CPUS=0-3,12-15 RAYON_NUM_THREADS=8 \
    python3 -c 'import collect_run_metadata as c, json; print(json.dumps(c.collect_affinity()))')"
python3 - "$OUT" <<'PY'
import json, sys
a = json.loads(sys.argv[1])
assert a["pinning"] == "taskset" and a["requested_measured_cpus"] == "4-11", a
assert a["observer_cpus"] == "0-3,12-15" and a["thread_budget"] == "8", a
for key in ("measured_cpus", "l3_domains", "cpu_boost", "scaling_governor"):
    assert key in a, key
PY
echo "PASS: zen5-cpu pinning table, per-row taskset, and affinity metadata"
