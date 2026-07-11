#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Fair VJP timing must include the forward pass that builds the VJP closure.
# Keeping `vjp(fn, x)` outside `runner` measures only the cached backward apply.
! rg -n '^ {12}_, vjp_fn = vjp\(fn, x\)' scripts/benchmark_gpu_linalg_ad.py
rg -n 'def runner\(\):' scripts/benchmark_gpu_linalg_ad.py
rg -n '^ {16}_, vjp_fn = vjp\(fn, x\)' scripts/benchmark_gpu_linalg_ad.py

echo "gpu_linalg_ad_runner_contract: ok"
