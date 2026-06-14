#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ids="$(uv run python scripts/suite_instances.py --suite-file benchmarks/cpu/einsum.yaml)"
test -n "$ids"
grep -q 'bin_matmul_256' <<< "$ids"
grep -q 'nary_matmul_chain_64' <<< "$ids"

subset="$(uv run python scripts/suite_instances.py --suite-file benchmarks/cpu/einsum.yaml --format lines | wc -l | tr -d ' ')"
test "$subset" -ge 18

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/selector.yaml" <<'YAML'
schema_version: 1
suite_id: cpu/einsum-smoke
title: Selector smoke
description: Nested selector smoke test.
defaults:
  run: {warmups: 1, runs: 1, timing_scope: steady_state_host_api}
  verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10}
backends: [tenferro-eager]
problems:
  source: benchmarks/cpu/einsum.yaml
  include: [bin_matmul_256]
YAML

filtered="$(uv run python scripts/suite_instances.py --suite-file "$TMP/selector.yaml")"
test "$filtered" = "bin_matmul_256"

BENCH_SUITE_INCLUDE="$ids" uv run python scripts/benchmark_python.py --backend pytorch --instance bin_matmul_256 --num-threads 1 >/tmp/bench_python_suite_filter.out
grep -q 'Loaded 1 instances' /tmp/bench_python_suite_filter.out

BENCH_SUITE_INCLUDE="missing_instance" uv run python scripts/benchmark_python.py --backend pytorch --num-threads 1 >/tmp/bench_python_empty.out 2>/tmp/bench_python_empty.err || true
grep -q 'No benchmark instances matched the suite selection' /tmp/bench_python_empty.err
