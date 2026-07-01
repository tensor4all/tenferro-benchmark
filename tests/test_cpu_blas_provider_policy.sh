#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

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
assert_eq system-mkl "$(normalize_cpu_blas_features system-mkl)" "darwin mkl"
assert_eq cpu-faer "$(normalize_cpu_blas_features cpu-faer)" "darwin faer"
assert_eq accelerate "$(blas_impl_for_features system-accelerate)" "accelerate blas metadata"
assert_eq mkl "$(blas_impl_for_features system-mkl)" "mkl blas metadata"
assert_eq none "$(blas_impl_for_features cpu-faer)" "faer blas metadata"

unset OPENBLAS_ROOT MKLROOT
ensure_blas_env_for_features system-accelerate
test -z "${OPENBLAS_ROOT:-}"

BENCHMARK_HOST_OS=Linux
assert_eq system-openblas "$(normalize_cpu_blas_features "")" "linux default"
assert_eq system-openblas "$(normalize_cpu_blas_features system-openblas)" "linux openblas"
assert_eq system-mkl "$(normalize_cpu_blas_features system-mkl)" "linux mkl"
assert_eq openblas "$(blas_impl_for_features system-openblas)" "openblas metadata"

if ensure_blas_env_for_features system-openblas 2>"$TMP/cpu_blas_provider_test.err"; then
  echo "linux openblas without OPENBLAS_ROOT should fail" >&2
  exit 1
fi

if ensure_blas_env_for_features system-mkl 2>"$TMP/cpu_blas_provider_test.err"; then
  echo "linux mkl without MKLROOT should fail" >&2
  exit 1
fi

mkl_root="$TMP/mkl"
mkdir -p "$mkl_root/lib" "$mkl_root/include"
MKLROOT="$mkl_root" ensure_blas_env_for_features system-mkl
assert_eq "$mkl_root" "$MKLROOT" "mkl root export"
case ":${LD_LIBRARY_PATH:-}:" in
  *":$mkl_root/lib:"*) ;;
  *) echo "MKL lib dir should be added to LD_LIBRARY_PATH" >&2; exit 1 ;;
esac
