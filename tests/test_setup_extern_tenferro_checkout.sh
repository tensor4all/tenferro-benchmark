#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git_config() {
  git -C "$1" config user.email "benchmark-test@example.com"
  git -C "$1" config user.name "Benchmark Test"
}

mkdir -p "$TMP/source"
git -C "$TMP/source" init -q
git_config "$TMP/source"
printf 'old\n' >"$TMP/source/marker.txt"
git -C "$TMP/source" add marker.txt
git -C "$TMP/source" commit -qm 'old tenferro'
OLD_COMMIT="$(git -C "$TMP/source" rev-parse HEAD)"
printf 'new\n' >"$TMP/source/marker.txt"
git -C "$TMP/source" commit -am 'new tenferro' -q
NEW_COMMIT="$(git -C "$TMP/source" rev-parse HEAD)"
git -C "$TMP/source" branch -M main
git -C "$TMP/source" clone --bare "$TMP/source" "$TMP/tenferro-rs.git" >/dev/null 2>&1

mkdir -p "$TMP/project-main/scripts"
cp "$ROOT/scripts/setup_extern_deps.sh" "$TMP/project-main/scripts/setup_extern_deps.sh"
cp "$ROOT/scripts/cpu_blas_provider.sh" "$TMP/project-main/scripts/cpu_blas_provider.sh"

(
  cd "$TMP/project-main"
  BENCHMARK_HOST_OS=Darwin \
  TENFERRO_REPO_URL="$TMP/tenferro-rs.git" \
  TENFERRO_REF=main \
    bash scripts/setup_extern_deps.sh >"$TMP/project-main.log" 2>&1

  test "$(git -C extern/tenferro-rs rev-parse HEAD)" = "$NEW_COMMIT"
  grep -q 'TENFERRO_CPU_FEATURES=system-accelerate' "$TMP/project-main.log"
  if grep -q 'OPENBLAS_ROOT=' "$TMP/project-main.log"; then
    echo "macOS Accelerate setup should not require or export OPENBLAS_ROOT" >&2
    cat "$TMP/project-main.log" >&2
    exit 1
  fi
)

mkdir -p "$TMP/project/scripts"
cp "$ROOT/scripts/setup_extern_deps.sh" "$TMP/project/scripts/setup_extern_deps.sh"
cp "$ROOT/scripts/cpu_blas_provider.sh" "$TMP/project/scripts/cpu_blas_provider.sh"

(
  cd "$TMP/project"
  export SETUP_EXTERN_DEPS_SKIP_MAIN=1
  export TENFERRO_REPO_URL="$TMP/tenferro-rs.git"
  export TENFERRO_REF=main

  # shellcheck source=/dev/null
  source scripts/setup_extern_deps.sh

  ensure_tenferro_checkout
  test "$(git -C extern/tenferro-rs rev-parse HEAD)" = "$NEW_COMMIT"

  git -C extern/tenferro-rs checkout -q --detach "$OLD_COMMIT"
  ensure_tenferro_checkout
  test "$(git -C extern/tenferro-rs rev-parse HEAD)" = "$OLD_COMMIT"

  git -C extern/tenferro-rs checkout -q --detach "$OLD_COMMIT"
  printf 'local\n' >extern/tenferro-rs/local-change.txt
  ensure_tenferro_checkout
  test "$(git -C extern/tenferro-rs rev-parse HEAD)" = "$OLD_COMMIT"
  test -f extern/tenferro-rs/local-change.txt
  rm extern/tenferro-rs/local-change.txt

  git -C extern/tenferro-rs checkout -q --detach "$OLD_COMMIT"
  printf 'dirty\n' >>extern/tenferro-rs/marker.txt
  ensure_tenferro_checkout
  test "$(git -C extern/tenferro-rs rev-parse HEAD)" = "$OLD_COMMIT"
  grep -q 'dirty' extern/tenferro-rs/marker.txt

  git -C extern/tenferro-rs checkout -- marker.txt
  export TENFERRO_UPDATE=1
  ensure_tenferro_checkout
  test "$(git -C extern/tenferro-rs rev-parse HEAD)" = "$NEW_COMMIT"
)

mkdir -p "$TMP/project-with-sibling/scripts"
cp "$ROOT/scripts/setup_extern_deps.sh" "$TMP/project-with-sibling/scripts/setup_extern_deps.sh"
cp "$ROOT/scripts/cpu_blas_provider.sh" "$TMP/project-with-sibling/scripts/cpu_blas_provider.sh"
git clone "$TMP/tenferro-rs.git" "$TMP/tenferro-rs" >/dev/null 2>&1
git -C "$TMP/tenferro-rs" checkout -q --detach "$OLD_COMMIT"

(
  cd "$TMP/project-with-sibling"
  export SETUP_EXTERN_DEPS_SKIP_MAIN=1
  export TENFERRO_REPO_URL="$TMP/tenferro-rs.git"
  export TENFERRO_REF=main

  # shellcheck source=/dev/null
  source scripts/setup_extern_deps.sh

  ensure_tenferro_checkout
  test -d "$TMP/tenferro-rs/.git"
  test "$(git -C "$TMP/tenferro-rs" rev-parse HEAD)" = "$OLD_COMMIT"
  test "$(git -C extern/tenferro-rs rev-parse HEAD)" = "$NEW_COMMIT"
)
