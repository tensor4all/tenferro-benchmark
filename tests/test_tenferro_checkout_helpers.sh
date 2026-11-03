#!/usr/bin/env bash
# The recorded commit is the checkout Cargo builds, and the upstream pull the
# collectors still do by default is skipped offline with GPU_BENCH_NO_PULL=1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git_config() {
    git -C "$1" config user.email "benchmark-test@example.com"
    git -C "$1" config user.name "Benchmark Test"
}

# shellcheck source=scripts/tenferro_checkout.sh
source scripts/tenferro_checkout.sh

mkdir -p "$TMP/repo"
git -C "$TMP/repo" init -q
git_config "$TMP/repo"
printf 'x\n' > "$TMP/repo/marker.txt"
git -C "$TMP/repo" add marker.txt
git -C "$TMP/repo" commit -qm init
git -C "$TMP/repo" branch -M main
COMMIT="$(git -C "$TMP/repo" rev-parse HEAD)"

test "$(tenferro_checkout_commit "$TMP/repo")" = "$COMMIT"
test -d "$TMP/repo/.git"

# A linked worktree records its commit through a .git FILE.
git -C "$TMP/repo" worktree add -q "$TMP/worktree" -b worktree-branch
test -f "$TMP/worktree/.git"
test "$(tenferro_checkout_commit "$TMP/worktree")" = "$COMMIT"

# An unreachable origin makes the default pull fail, and the offline opt-out skip it.
git -C "$TMP/repo" remote add origin "$TMP/does-not-exist.git"
status=0
update_tenferro_checkout_if_main "$TMP/repo" >"$TMP/pull.log" 2>&1 || status=$?
test "$status" != 0
GPU_BENCH_NO_PULL=1 update_tenferro_checkout_if_main "$TMP/repo"
test "$(tenferro_checkout_commit "$TMP/repo")" = "$COMMIT"
# A non-main checkout is never pulled, even without the opt-out.
update_tenferro_checkout_if_main "$TMP/worktree"
test "$(tenferro_checkout_commit "$TMP/worktree")" = "$COMMIT"

# Collectors describe extern/tenferro-rs, the path Cargo resolves; TENFERRO_RS_DIR
# is the setup clone location and must not reach the recorded metadata.
for script in scripts/run_gpu_linalg_jvp_vjp.sh scripts/run_gpu_throughput.sh \
              scripts/run_gpu_permutation.sh scripts/run_gpu_suite.sh; do
    grep -q 'extern/tenferro-rs' "$script"
    if grep -q 'TENFERRO_RS_DIR' "$script"; then
        echo "$script records TENFERRO_RS_DIR instead of the checkout Cargo builds" >&2
        exit 1
    fi
done
for script in scripts/run_gpu_linalg_jvp_vjp.sh scripts/run_gpu_public_api.sh \
              scripts/run_gpu_throughput.sh scripts/run_cpu_followup.sh \
              scripts/run_gpu_suite.sh; do
    grep -q 'GPU_BENCH_NO_PULL\|update_tenferro_checkout_if_main' "$script"
done

# run_gpu_suite.sh had its own copy of both concerns; the helper is now the only
# path, so the offline opt-out covers the routine entry point too.
if grep -q 'pull --ff-only' scripts/run_gpu_suite.sh; then
    echo "run_gpu_suite.sh still pulls on its own" >&2
    exit 1
fi
if grep -q '^resolve_git_commit()' scripts/run_gpu_suite.sh; then
    echo "run_gpu_suite.sh keeps a duplicate commit resolver" >&2
    exit 1
fi

echo "tenferro checkout helpers ok"
