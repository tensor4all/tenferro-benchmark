#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# shellcheck source=/dev/null
source "$ROOT/scripts/cpu_blas_provider.sh"

assert_eq() {
  local expected="$1"
  local actual="$2"
  local label="$3"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label: expected '$expected', got '$actual'" >&2
    exit 1
  fi
}

BENCHMARK_HOST_OS=Darwin
assert_eq system-accelerate "$(normalize_cpu_blas_features "")" "darwin default"
assert_eq system-accelerate "$(normalize_cpu_blas_features system-openblas)" "darwin openblas override"
assert_eq system-accelerate "$(normalize_cpu_blas_features system-accelerate)" "darwin accelerate"
assert_eq cpu-faer "$(normalize_cpu_blas_features cpu-faer)" "darwin faer"
assert_eq accelerate "$(blas_impl_for_features system-accelerate)" "accelerate blas metadata"
assert_eq none "$(blas_impl_for_features cpu-faer)" "faer blas metadata"

unset OPENBLAS_ROOT
ensure_blas_env_for_features system-accelerate
test -z "${OPENBLAS_ROOT:-}"

BENCHMARK_HOST_OS=Linux
assert_eq system-openblas "$(normalize_cpu_blas_features "")" "linux default"
assert_eq system-openblas "$(normalize_cpu_blas_features system-openblas)" "linux openblas"
assert_eq openblas "$(blas_impl_for_features system-openblas)" "openblas metadata"

if ensure_blas_env_for_features system-openblas 2>"$ROOT/.cpu_blas_provider_test.err"; then
  echo "linux openblas without OPENBLAS_ROOT should fail" >&2
  exit 1
fi
rm -f "$ROOT/.cpu_blas_provider_test.err"
