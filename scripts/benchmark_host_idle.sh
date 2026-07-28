#!/usr/bin/env bash

# Refuse to start timing collection while another compiler, test run, or
# benchmark is already consuming the host. Set BENCHMARK_ALLOW_BUSY_HOST=1 only
# for intentional diagnostics whose numbers will not be published.
assert_benchmark_host_idle() {
    if [[ "${BENCHMARK_ALLOW_BUSY_HOST:-0}" == "1" ]]; then
        return 0
    fi

    local busy
    busy="$({
        ps -axo pid=,command= 2>/dev/null || true
    } | grep -E \
        '[r]ustc |cargo-nextest nextest|cargo (nextest|test|bench)( |$)|premerge_gate\.sh|metamorphic_equivalence\.sh|target/(release|debug)/benchmark_|python[^ ]* .*benchmark_.*\.py' \
        | grep -v ' grep -E ' \
        | head -20 \
        | cut -c1-240 \
        || true)"
    if [[ -n "$busy" ]]; then
        echo "ERROR: refusing to collect benchmark timings on a busy host." >&2
        echo "Competing compiler/test/benchmark processes:" >&2
        echo "$busy" >&2
        echo "Wait for them to finish, or set BENCHMARK_ALLOW_BUSY_HOST=1 for non-publication diagnostics." >&2
        return 1
    fi
}
