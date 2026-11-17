#!/usr/bin/env bash
# Build harness binaries against one tenferro-rs checkout into a target
# directory keyed by that checkout's commit, so baseline and candidate builds
# never share (and silently reuse) compiled tenferro-rs artifacts.
#
# Usage: scripts/build_for_tenferro_rev.sh <tenferro-rs-dir> <out-var-file> [bin...]
#
# Builds the existing real checkout in place, or re-points its symlink at
# <tenferro-rs-dir> and restores the previous link target afterwards. Writes shell assignments to
# <out-var-file>: TENFERRO_REV, TENFERRO_DIRTY, TENFERRO_DIR, BIN_DIR.
# Default bins: benchmark_cpu_session (the cross-revision binary). The build
# uses the harness's one CPU feature set (`native`) and honors
# BENCH_BUILD_PROFILE (default release). It clears RUSTC_WRAPPER so a compiler
# cache cannot mix artifacts across tenferro-rs revisions.
#
# tenferro-rs 0.12 has one CPU backend (#2608): there is no vendor BLAS/LAPACK
# provider to select, so this build links nothing but the crate's own kernels.
# Build a revision older than that with the matching older harness revision.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TENFERRO_DIR="$(cd "${1:?tenferro-rs checkout required}" && pwd -P)"
OUT_FILE="${2:?output variable file required}"
shift 2
BINS=("${@:-benchmark_cpu_session}")
# shellcheck disable=SC2206
BINS=(${BINS[*]})
if [[ ! -d "$TENFERRO_DIR/crates/tenferro-cpu" ]]; then
    echo "ERROR: $TENFERRO_DIR is not a tenferro-rs checkout" >&2
    exit 1
fi
FEATURES=native
PROFILE="${BENCH_BUILD_PROFILE:-release}"
LINK="$PROJECT_DIR/extern/tenferro-rs"

rev="$(git -C "$TENFERRO_DIR" rev-parse HEAD)"
dirty=false
[[ -z "$(git -C "$TENFERRO_DIR" status --porcelain --untracked-files=no)" ]] || dirty=true
suffix=""
if [[ $dirty == true ]]; then suffix="-dirty"; fi
target="$PROJECT_DIR/target/tenferro-rev/${rev:0:12}${suffix}"

in_place=false
if [[ -d "$LINK" && ! -L "$LINK" && "$(cd "$LINK" && pwd -P)" == "$TENFERRO_DIR" ]]; then
    # Normal setup creates a real checkout. Building that same revision does
    # not require replacing it with a symlink.
    in_place=true
elif [[ -e "$LINK" && ! -L "$LINK" ]]; then
    echo "ERROR: $LINK is a real directory; this script only re-points a symlink." >&2
    exit 1
fi
previous=""
[[ -L "$LINK" ]] && previous="$(readlink "$LINK")"
restore() { if [[ -n "$previous" ]]; then ln -sfn "$previous" "$LINK"; fi; }
trap restore EXIT
if [[ "$in_place" != true ]]; then
    ln -sfn "$TENFERRO_DIR" "$LINK"
fi

bin_args=()
for bin in "${BINS[@]}"; do bin_args+=(--bin "$bin"); done
profile_flag=()
subdir=debug
if [[ "$PROFILE" == release ]]; then profile_flag=(--release); subdir=release; fi
RUSTC_WRAPPER= CARGO_TARGET_DIR="$target" cargo build -j 16 "${profile_flag[@]+"${profile_flag[@]}"}" \
    --no-default-features --features native "${bin_args[@]+"${bin_args[@]}"}" >&2

bin_dir="$target/$subdir"
{
    printf 'TENFERRO_REV=%q\n' "$rev"
    printf 'TENFERRO_DIRTY=%q\n' "$dirty"
    printf 'TENFERRO_DIR=%q\n' "$TENFERRO_DIR"
    printf 'BIN_DIR=%q\n' "$bin_dir"
    printf 'BUILD_FEATURES=%q\n' "$FEATURES"
} > "$OUT_FILE"
echo "built ${BINS[*]} for tenferro-rs $rev (dirty=$dirty) in $bin_dir" >&2
