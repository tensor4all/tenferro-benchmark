#!/usr/bin/env bash
# Build harness binaries against one tenferro-rs checkout into a target
# directory keyed by that checkout's commit, so baseline and candidate builds
# never share (and silently reuse) compiled tenferro-rs artifacts.
#
# Usage: scripts/build_for_tenferro_rev.sh <tenferro-rs-dir> <out-var-file> [bin...]
#
# Re-points extern/tenferro-rs at <tenferro-rs-dir> for the build and restores
# the previous link target afterwards. Writes shell assignments to
# <out-var-file>: TENFERRO_REV, TENFERRO_DIRTY, TENFERRO_DIR, BIN_DIR.
# Default bins: benchmark_cpu_session cpu_route_diagnostic (the cross-revision
# binaries). Honors TENFERRO_CPU_FEATURES (default cpu-faer) and
# BENCH_BUILD_PROFILE (default release).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TENFERRO_DIR="$(cd "${1:?tenferro-rs checkout required}" && pwd -P)"
OUT_FILE="${2:?output variable file required}"
shift 2
BINS=("${@:-benchmark_cpu_session cpu_route_diagnostic}")
# shellcheck disable=SC2206
BINS=(${BINS[*]})
if [[ ! -d "$TENFERRO_DIR/crates/tenferro-cpu" ]]; then
    echo "ERROR: $TENFERRO_DIR is not a tenferro-rs checkout" >&2
    exit 1
fi
FEATURES="${TENFERRO_CPU_FEATURES:-cpu-faer}"
PROFILE="${BENCH_BUILD_PROFILE:-release}"
LINK="$PROJECT_DIR/extern/tenferro-rs"

rev="$(git -C "$TENFERRO_DIR" rev-parse HEAD)"
dirty=false
[[ -z "$(git -C "$TENFERRO_DIR" status --porcelain --untracked-files=no)" ]] || dirty=true
suffix=""
if [[ $dirty == true ]]; then suffix="-dirty"; fi
target="$PROJECT_DIR/target/tenferro-rev/${rev:0:12}${suffix}"

if [[ -e "$LINK" && ! -L "$LINK" ]]; then
    echo "ERROR: $LINK is a real directory; this script only re-points a symlink." >&2
    exit 1
fi
previous=""
[[ -L "$LINK" ]] && previous="$(readlink "$LINK")"
restore() { if [[ -n "$previous" ]]; then ln -sfn "$previous" "$LINK"; fi; }
trap restore EXIT
ln -sfn "$TENFERRO_DIR" "$LINK"

bin_args=()
for bin in "${BINS[@]}"; do bin_args+=(--bin "$bin"); done
profile_flag=()
subdir=debug
if [[ "$PROFILE" == release ]]; then profile_flag=(--release); subdir=release; fi
RUSTC_WRAPPER= CARGO_TARGET_DIR="$target" cargo build -j 16 "${profile_flag[@]}" \
    --no-default-features --features "$FEATURES" "${bin_args[@]}" >&2

bin_dir="$target/$subdir"
{
    printf 'TENFERRO_REV=%q\n' "$rev"
    printf 'TENFERRO_DIRTY=%q\n' "$dirty"
    printf 'TENFERRO_DIR=%q\n' "$TENFERRO_DIR"
    printf 'BIN_DIR=%q\n' "$bin_dir"
    printf 'BUILD_FEATURES=%q\n' "$FEATURES"
} > "$OUT_FILE"
echo "built ${BINS[*]} for tenferro-rs $rev (dirty=$dirty) in $bin_dir" >&2
