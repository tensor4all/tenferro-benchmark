#!/usr/bin/env bash

# Per-thread-row CPU affinity for target profiles that pin their measured process.
#
# A profile with a pinning table (today only `zen5-cpu`) runs every thread row
# under a fixed `taskset` mask and records the masks in run metadata. Other
# profiles are unpinned and unchanged.
#
# zen5-cpu is the AMD Ryzen AI 9 HX 470 workstation: 12 physical cores, SMT2,
# two L3 domains (cores 0-3 with SMT siblings 12-15, cores 4-11 with SMT
# siblings 16-23). The measured set uses one logical CPU per physical core so
# SMT never inflates a row, and the observer (whatever else is running, e.g. the
# agent that drives the run) stays on the other L3 domain, never on an SMT
# sibling of a measured core:
#
#   threads  measured (taskset -c)  observer (taskset -c)
#   1        0                      4-11,16-23
#   4        0-3                    4-11,16-23
#   8        4-11                   0-3,12-15
#
# BENCHMARK_OBSERVER_FOLLOW=1 additionally moves the runner's ancestor processes (the
# shell, terminal multiplexer or agent that launched it) to the row's observer set at
# every row, so a single multi-row invocation (`run_all.sh 1 4 8`) honours the table even
# though the 8T observer set differs from the 1T/4T one. It only reaches processes in the
# runner's PID namespace: from inside the devcontainer it cannot move a host agent, which
# then has to be moved by hand (see the handoff).
#
# BENCHMARK_OBSERVER_HANDSHAKE=<dir> covers that case. At every row the runner writes the
# row's observer mask to <dir>/request and waits (up to 120 s) until <dir>/ack echoes the
# request, before the idle check and the timings. scripts/observer_follow_host.sh, run on
# the host, answers each request by moving the given host processes to the mask. The
# directory must be the same files on both sides (a path under the bind-mounted checkout).
# Note that the rows interleave: `run_all.sh 1 4 8` runs 1, 4, 8 for every suite in turn.
#
# BENCHMARK_CPU_AFFINITY overrides the measured mask for diagnostics only; the
# override is recorded, and a run that sets it is not a publication run.

# Echo the measured CPU mask for PROFILE and THREADS, or nothing when the
# profile is not pinned. Return 2 for a pinned profile without a row for THREADS.
cpu_pinning_measured_mask() {
    local profile="${1:?profile required}" threads="${2:?threads required}"
    case "$profile" in
        zen5-cpu)
            case "$threads" in
                1) printf '%s\n' "0" ;;
                4) printf '%s\n' "0-3" ;;
                8) printf '%s\n' "4-11" ;;
                *) return 2 ;;
            esac
            ;;
        *) return 0 ;;
    esac
}

# Echo the observer CPU mask for PROFILE and THREADS (same contract).
cpu_pinning_observer_mask() {
    local profile="${1:?profile required}" threads="${2:?threads required}"
    case "$profile" in
        zen5-cpu)
            case "$threads" in
                1|4) printf '%s\n' "4-11,16-23" ;;
                8) printf '%s\n' "0-3,12-15" ;;
                *) return 2 ;;
            esac
            ;;
        *) return 0 ;;
    esac
}

# Move every ancestor of this shell (and its threads) to the observer mask.
cpu_pinning_move_ancestors() {
    local mask="${1:?mask required}" pid ppid
    pid="$$"
    while :; do
        ppid="$(awk '{print $4}' "/proc/$pid/stat" 2>/dev/null)" || break
        [[ -n "$ppid" && "$ppid" -gt 1 ]] || break
        taskset -apc "$mask" "$ppid" >/dev/null 2>&1 \
            || echo "WARNING: could not move ancestor $ppid to observer CPUs $mask" >&2
        pid="$ppid"
    done
}

# Ask the host helper in DIR to move the observer to MASK and wait for its ack.
cpu_pinning_handshake() {
    local dir="${1:?dir required}" mask="${2:?mask required}" token waited=0
    mkdir -p "$dir"
    token="$$-$(date +%s%N) $mask"
    printf '%s\n' "$token" > "$dir/request.tmp" && mv "$dir/request.tmp" "$dir/request"
    until [[ "$(cat "$dir/ack" 2>/dev/null)" == "$token" ]]; do
        if (( waited >= 1200 )); then
            echo "ERROR: no observer ack for '$mask' in $dir after 120 s (is scripts/observer_follow_host.sh running?)." >&2
            return 1
        fi
        sleep 0.1
        waited=$((waited + 1))
    done
}

# Pin the current shell (and so every process it starts afterwards) to the
# measured mask of the active profile's THREADS row, and export the recorded
# masks for collect_run_metadata.py. No-op for unpinned profiles.
apply_cpu_affinity() {
    local threads="${1:?threads required}"
    local profile="${BENCHMARK_TARGET_PROFILE:-}"
    local mask observer status=0
    [[ -n "$profile" ]] || return 0
    mask="$(cpu_pinning_measured_mask "$profile" "$threads")" || status=$?
    if (( status == 2 )); then
        echo "ERROR: profile $profile has no pinning row for $threads threads (rows: 1, 4, 8)." >&2
        return 1
    fi
    unset BENCHMARK_CPU_PINNING BENCHMARK_MEASURED_CPUS BENCHMARK_OBSERVER_CPUS BENCHMARK_CPU_AFFINITY_OVERRIDE
    [[ -n "$mask" ]] || return 0
    observer="$(cpu_pinning_observer_mask "$profile" "$threads")"
    if [[ -n "${BENCHMARK_CPU_AFFINITY:-}" ]]; then
        export BENCHMARK_CPU_AFFINITY_OVERRIDE="$mask"
        mask="$BENCHMARK_CPU_AFFINITY"
        echo "WARNING: BENCHMARK_CPU_AFFINITY=$mask overrides the $profile row for $threads threads; diagnostics only." >&2
    fi
    if ! command -v taskset >/dev/null 2>&1; then
        echo "ERROR: taskset is required to pin $profile rows (util-linux)." >&2
        return 1
    fi
    taskset -pc "$mask" "$$" >/dev/null || {
        echo "ERROR: could not pin the runner to CPUs $mask (cgroup/cpuset too narrow?)." >&2
        return 1
    }
    if [[ "${BENCHMARK_OBSERVER_FOLLOW:-0}" == "1" ]]; then
        cpu_pinning_move_ancestors "$observer"
    fi
    if [[ -n "${BENCHMARK_OBSERVER_HANDSHAKE:-}" ]]; then
        cpu_pinning_handshake "$BENCHMARK_OBSERVER_HANDSHAKE" "$observer" || return 1
    fi
    export BENCHMARK_CPU_PINNING=taskset
    export BENCHMARK_MEASURED_CPUS="$mask"
    export BENCHMARK_OBSERVER_CPUS="$observer"
}
