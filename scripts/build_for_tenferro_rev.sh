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
# Default bins: benchmark_cpu_session (the cross-revision binary). Honors
# TENFERRO_CPU_FEATURES (default native; BLAS features need their provider
# environment, e.g. OPENBLAS_ROOT/MKLROOT) and BENCH_BUILD_PROFILE (default
# release). The build clears RUSTC_WRAPPER so a compiler cache cannot mix
# artifacts across tenferro-rs revisions.
#
# Feature names follow tenferro-rs #2004. Revisions before that change used the
# old harness/feature names (cpu-faer/cpu-blas/system-*); build such a revision
# with the matching older harness revision instead of this one.
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
# shellcheck source=scripts/cpu_blas_provider.sh
source "$SCRIPT_DIR/cpu_blas_provider.sh"
FEATURES="${TENFERRO_CPU_FEATURES:-native}"
ensure_blas_env_for_features "$FEATURES"
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
