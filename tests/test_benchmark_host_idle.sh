#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/benchmark_host_idle.sh"
unset BENCHMARK_ALLOW_BUSY_HOST

# Mock only this test shell's process listing; never stop a real process.
ps() { printf '123 %s\n' "$listed_command"; }
for listed_command in \
    '/toolchain/bin/rustc --crate-name example' \
    '/toolchain/bin/clippy-driver --crate-name example' \
    '/toolchain/bin/cargo-clippy clippy --workspace' \
    '/toolchain/bin/cargo clippy --workspace'; do
    if assert_benchmark_host_idle >/dev/null 2>&1; then
        echo "FAIL: idle guard admitted $listed_command" >&2
        exit 1
    fi
done
listed_command='/usr/bin/sleep infinity'
assert_benchmark_host_idle

# A pinned row also needs its measured CPUs idle, seen host-wide through /proc/stat.
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
export BENCHMARK_PROC_STAT="$TMP/stat"
# Fields after the name: user nice system idle. The mocked sleep swaps in the
# second snapshot: cpu4 spends the whole window busy, cpu5 stays idle.
sleep() { cp "$TMP/next" "$BENCHMARK_PROC_STAT"; }
printf 'cpu4 0 0 0 100 0 0 0 0 0 0\ncpu5 0 0 0 100 0 0 0 0 0 0\n' > "$BENCHMARK_PROC_STAT"
printf 'cpu4 100 0 0 100 0 0 0 0 0 0\ncpu5 0 0 0 200 0 0 0 0 0 0\n' > "$TMP/next"
export BENCHMARK_MEASURED_CPUS=4
if assert_benchmark_host_idle >/dev/null 2>&1; then
    echo "FAIL: idle guard admitted a busy measured CPU" >&2
    exit 1
fi
export BENCHMARK_MEASURED_CPUS=5
assert_benchmark_host_idle
unset BENCHMARK_MEASURED_CPUS BENCHMARK_PROC_STAT
unset -f sleep
echo 'PASS: active compilers are rejected; an idle host is admitted; busy measured CPUs are rejected'
