#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin" "$TMP/openblas" "$TMP/strided-rs-benchmark-suite"

cat > "$TMP/bin/brew" <<SH
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" == "--prefix" && "\${2:-}" == "openblas" ]]; then
  echo "$TMP/openblas"
else
  exit 1
fi
SH

cat > "$TMP/bin/cargo" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${CARGO_CALL_LOG:?}"
if [[ "${1:-}" == "build" ]]; then
  exit 0
fi
if [[ "${1:-}" == "run" ]]; then
  grep -q -- '--bin tenferro-einsum-benchmark' <<< "$*" || {
    echo "cargo run missing --bin tenferro-einsum-benchmark: $*" >&2
    exit 42
  }
  printf 'tenferro-%s\nBackend: tenferro-%s\nStrategy: opt_flops\nInstance Tensors log10FLOPS log2SIZE Median (ms) IQR (ms) Compile (ms)\nbin_matmul_256 2 0 0 1.0 0.1 0.0\n' "${TENFERRO_MODE:-trace}" "${TENFERRO_MODE:-trace}"
  exit 0
fi
exit 0
SH
chmod +x "$TMP/bin/brew" "$TMP/bin/cargo"

export CARGO_CALL_LOG="$TMP/cargo.log"
(
  cd "$ROOT"
  PATH="$TMP/bin:/usr/bin:/bin" BENCHMARK_TIMESTAMP=20000101_000000 ./scripts/run_all_rust.sh 1 >/tmp/run_all_rust_bin_test.out
)

grep -q -- '--bin tenferro-einsum-benchmark' "$CARGO_CALL_LOG"
