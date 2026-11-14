#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# The benchmark pod carries no Rust toolchain: every collector must use the
# binaries the hosted build job ships. The throughput lane missed that and ran
# `cargo build` on the pod (run 43549221099: "cargo: command not found", exit
# 127), losing the whole collection. This pins the branch that keeps it out.
python3 - "$ROOT/scripts/run_gpu_throughput.sh" <<'PY'
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
start = text.index('if [[ "${GPU_BENCH_PREBUILT_RUST:-0}" == "1" ]]; then')
block = text[start:text.index('\nfi', start)]
assert 'gpu_artifact.py' in block, 'prebuilt branch does not verify the hosted binary'
for binary in ('benchmark_gpu_rust', 'benchmark_gpu_linalg_ad'):
    assert binary in block, f'prebuilt branch does not install {binary}'
assert block.index('else') < block.rindex('cargo build'), \
    'the cargo build must stay in the non-prebuilt branch only'
PY

echo "gpu throughput runner contract: ok"
