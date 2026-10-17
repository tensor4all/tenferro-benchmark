#!/usr/bin/env bash
# Maintained report/summary publication contract.
#
# A maintained report must publish the formatter's dashboard summary sidecar
# next to the Markdown, not leave it only in the raw run directory. This guards
# the publication step directly (the end-to-end GPU run is asserted in
# tests/test_gpu_benchmark_contract.sh on a CUDA host).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# The standard GPU runner copies the raw report into result/, so it must copy
# the sidecar alongside it.
if ! rg -q 'cp "\$\{MARKDOWN_OUT%\.md\}\.summary\.json" "\$\{REPORT_OUT%\.md\}\.summary\.json"' scripts/run_gpu_suite.sh; then
    echo "run_gpu_suite.sh does not publish the maintained summary sidecar" >&2
    exit 1
fi

# Functional: raw report + sidecar -> both next to the maintained Markdown.
mkdir -p "$TMP/raw" "$TMP/result/nvidia-gpu/gpu"
printf 'report\n' > "$TMP/raw/report.md"
printf '{"suite_id": "gpu/dense", "rows": []}\n' > "$TMP/raw/report.summary.json"
raw="$TMP/raw/report.md"
maintained="$TMP/result/nvidia-gpu/gpu/dense.md"
cp "$raw" "$maintained"
if [[ -f "${raw%.md}.summary.json" ]]; then
    cp "${raw%.md}.summary.json" "${maintained%.md}.summary.json"
fi
test -s "$maintained"
test -s "$TMP/result/nvidia-gpu/gpu/dense.summary.json"
cmp "$raw" "$maintained"
cmp "${raw%.md}.summary.json" "${maintained%.md}.summary.json"

# A raw report without a sidecar still publishes the Markdown and does not
# create a stale summary.
printf 'report\n' > "$TMP/raw2.md"
maintained2="$TMP/result/nvidia-gpu/gpu/einsum.md"
cp "$TMP/raw2.md" "$maintained2"
test -s "$maintained2"
test ! -e "${maintained2%.md}.summary.json"

echo "maintained report summary publication contract passed"
