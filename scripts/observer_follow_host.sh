#!/usr/bin/env bash
# Answer the runner's observer requests (BENCHMARK_OBSERVER_HANDSHAKE, see cpu_pinning.sh)
# from the host, for a run inside the devcontainer that cannot reach host processes.
#
#   scripts/observer_follow_host.sh <dir> <pid>...
#
# For every new <dir>/request ("<token> <mask>") it moves each pid, with all its threads,
# to <mask> and then writes the request back to <dir>/ack. Run it on the observer side
# itself; it moves its own shell along with the pids.
set -euo pipefail
dir="${1:?usage: observer_follow_host.sh <dir> <pid>...}"
shift
(( $# > 0 )) || { echo "usage: observer_follow_host.sh <dir> <pid>..." >&2; exit 2; }
mkdir -p "$dir"
last=""
while :; do
    request="$(cat "$dir/request" 2>/dev/null || true)"
    if [[ -n "$request" && "$request" != "$last" ]]; then
        mask="${request##* }"
        for pid in "$@" "$$"; do
            taskset -apc "$mask" "$pid" >/dev/null \
                || echo "WARNING: could not move $pid to $mask" >&2
        done
        printf '%s\n' "$request" > "$dir/ack.tmp" && mv "$dir/ack.tmp" "$dir/ack"
        echo "$(date -u +%T) observer -> $mask"
        last="$request"
    fi
    sleep 0.2
done
