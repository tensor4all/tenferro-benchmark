#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
import json
from math import prod
from pathlib import Path

import yaml

from scripts.benchmark_layout import safe_target_profile

assert safe_target_profile("mac-gpu") == "mac-gpu"

suite = yaml.safe_load(Path("benchmarks/gpu/permutation-mac.yaml").read_text())
assert suite["suite_id"] == "gpu/permutation"
assert suite["defaults"]["device"]["kind"] == "metal"
assert suite["backends"] == [
    "tenferro-webgpu-transpose-baseline",
    "tenferro-webgpu-to-contiguous",
    "pytorch-mps",
    "jax-metal",
    "memcpy-metal-d2d",
]

document = json.loads(
    Path("data/instances/gpu_permutation_mac_patterns.json").read_text()
)
patterns = {pattern["id"]: pattern for pattern in document["patterns"]}
assert set(suite["problems"]["include"]) == set(patterns)
assert len(patterns) == 10
for pattern in patterns.values():
    assert pattern["dtype"] == "f32"
    assert prod(pattern["shape"]) >= 2**20
    participants = set(pattern["participants_gpu"])
    if pattern["id"] == "mac_memcpy_1d":
        assert participants == {"memcpy-metal-d2d"}
    else:
        assert "tenferro-webgpu-to-contiguous" in participants
        assert "pytorch-mps" in participants
        assert "jax-metal" in participants

metal_runner = Path("scripts/benchmark_gpu_permutation_metal.py").read_text()
assert '"per_call_allocation": backend != "memcpy-metal-d2d"' in metal_runner
assert "fresh compact column-major destination per timed call" in metal_runner

formatter = Path("scripts/format_gpu_permutation_results.py").read_text()
specification = Path("docs/gpu-permutation-suite.md").read_text()
allocation_contract = "Both tenferro and PyTorch MPS allocate a fresh destination"
assert allocation_contract in formatter
assert allocation_contract in specification
PY

rg -q 'status="not_configured"' scripts/benchmark_gpu_permutation_metal.py
rg -q 'platform.*metal' scripts/benchmark_gpu_permutation_metal.py
rg -q 'torch\.mps\.synchronize' scripts/benchmark_gpu_permutation_metal.py
rg -q 'backend\.synchronize' src/bin/benchmark_gpu_permutation_webgpu.rs
rg -q 'tenferro_revision' src/bin/benchmark_gpu_permutation_webgpu.rs
rg -q 'Apple M5 Max' docs/gpu-permutation-suite.md

echo "mac-gpu permutation profile contract passed"
