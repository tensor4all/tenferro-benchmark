#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

uv run python - <<'PY'
import json
from math import prod
from pathlib import Path

import yaml

suite = yaml.safe_load(Path("benchmarks/gpu/permutation.yaml").read_text())
source = suite["problems"]["source"]
assert source == "data/instances/gpu_permutation_patterns.json"
assert source != "data/instances/permutation_patterns.json"

document = json.loads(Path(source).read_text())
patterns = {pattern["id"]: pattern for pattern in document["patterns"]}
included = suite["problems"]["include"]
assert set(included) == set(patterns)

for pattern_id in included:
    pattern = patterns[pattern_id]
    elems = prod(pattern["shape"])
    assert elems >= 3**18, f"{pattern_id} is too small for the GPU timing suite"
    assert len(pattern["shape"]) <= 25, f"{pattern_id} exceeds PyTorch's rank limit"
    participants = set(pattern["participants_gpu"])
    if participants == {"memcpy-d2d"}:
        continue
    assert "tenferro-cuda-to-contiguous" in participants
    assert "pytorch-cuda" in participants
PY
