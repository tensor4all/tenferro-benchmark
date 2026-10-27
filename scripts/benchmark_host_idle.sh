#!/usr/bin/env bash

# Refuse to start timing collection while another compiler, test run, or
# benchmark is already consuming the host. Set BENCHMARK_ALLOW_BUSY_HOST=1 only
# for intentional diagnostics whose numbers will not be published.
# Pinned profiles (zen5-cpu) also require the measured CPUs themselves to be idle.
# The process-name scan below only sees this PID namespace, so inside the
# devcontainer it cannot see a build or test running on the host; /proc/stat is
# host-wide, so a busy measured core is caught either way. The window and busy
# threshold are BENCHMARK_IDLE_WINDOW_SECONDS (default 1) and
# BENCHMARK_IDLE_MAX_BUSY (default 0.05 = 5% of a core, averaged over the set).
# Only active when the runner pinned the row (BENCHMARK_MEASURED_CPUS is set).
benchmark_measured_cpus_busy_fraction() {
    local cpus="$1" window="${2:-1}" stat="${BENCHMARK_PROC_STAT:-/proc/stat}"
    local first second
    first="$(cat "$stat")"
    sleep "$window"
    second="$(cat "$stat")"
    awk -v cpus="$cpus" '
        function expand(list,   parts, n, i, range, lo, hi, c) {
            n = split(list, parts, ",")
            for (i = 1; i <= n; i++) {
                if (split(parts[i], range, "-") == 2) { lo = range[1]; hi = range[2] }
                else { lo = hi = parts[i] }
                for (c = lo; c <= hi; c++) want["cpu" c] = 1
            }
        }
        BEGIN { expand(cpus) }
        /^cpu[0-9]+ / {
            total = 0
            for (i = 2; i <= NF; i++) total += $i
            idle = $5 + $6
            if (FILENAME == ARGV[1]) { t0[$1] = total; i0[$1] = idle }
            else if (($1 in want) && ($1 in t0)) {
                dt = total - t0[$1]; di = idle - i0[$1]
                if (dt > 0) { busy += (dt - di) / dt }
                count++
            }
        }
        END { if (count > 0) printf "%.4f\n", busy / count; else print "0" }
    ' <(printf '%s\n' "$first") <(printf '%s\n' "$second")
}

assert_benchmark_host_idle() {
    if [[ "${BENCHMARK_ALLOW_BUSY_HOST:-0}" == "1" ]]; then
        return 0
    fi

    if [[ -n "${BENCHMARK_MEASURED_CPUS:-}" ]]; then
        local busy_fraction max_busy="${BENCHMARK_IDLE_MAX_BUSY:-0.05}"
        busy_fraction="$(benchmark_measured_cpus_busy_fraction "$BENCHMARK_MEASURED_CPUS" "${BENCHMARK_IDLE_WINDOW_SECONDS:-1}")"
        if awk -v b="$busy_fraction" -v m="$max_busy" 'BEGIN { exit !(b > m) }'; then
            echo "ERROR: refusing to collect benchmark timings: measured CPUs $BENCHMARK_MEASURED_CPUS were ${busy_fraction} busy (limit $max_busy)." >&2
            echo "Something else runs on the measured cores (this scan is host-wide, also from inside a container)." >&2
            echo "Wait for it to finish, or set BENCHMARK_ALLOW_BUSY_HOST=1 for non-publication diagnostics." >&2
            return 1
        fi
    fi

    local busy
    busy="$({
        ps -axo pid=,command= 2>/dev/null || true
    } | grep -E \
        '[r]ustc |[c]lippy-driver |cargo-clippy |cargo-nextest nextest|cargo (nextest|test|bench|clippy)( |$)|premerge_gate\.sh|metamorphic_equivalence\.sh|target/(release|debug)/benchmark_|python[^ ]* .*benchmark_.*\.py' \
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
