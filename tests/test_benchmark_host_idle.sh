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
echo 'PASS: active compilers are rejected; an idle host is admitted'
