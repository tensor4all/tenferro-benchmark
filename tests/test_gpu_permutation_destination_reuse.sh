#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if command -v uv >/dev/null 2>&1; then
    PYTHON=(uv run python)
else
    PYTHON=(python3)
fi

"${PYTHON[@]}" - "$ROOT" "$TMP" <<'PY'
import json
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
tmp = Path(sys.argv[2])

suite = yaml.safe_load((root / "benchmarks/gpu/permutation.yaml").read_text())
reuse = "tenferro-cuda-destination-reuse"
assert reuse in suite["backends"]

patterns = json.loads(
    (root / "data/instances/gpu_permutation_patterns.json").read_text()
)["patterns"]
for pattern in patterns:
    participants = set(pattern["participants_gpu"])
    if pattern["src_layout"]["kind"] == "col_major" and "cutensor" in participants:
        assert reuse in participants, pattern["id"]
    else:
        assert reuse not in participants, pattern["id"]

records = [
    {
        "schema_version": 1,
        "suite_id": "gpu/permutation",
        "runner": "rust",
        "pattern_id": "transpose_2d_32768_16384",
        "label": "2D transpose",
        "backend": backend,
        "shape": [32768, 16384],
        "perm": [1, 0],
        "dtype": "f64",
        "elems": 536870912,
        "bytes_rw": 8589934592,
        "device": "A100",
        "status": "ok",
        "correctness": "passed",
        "per_call_allocation": allocation,
        "warmup": 1,
        "iters": 1,
        "median_ms": median,
        "p25_ms": median,
        "p75_ms": median,
        "bandwidth_gbs": 1000.0,
        "notes": notes,
    }
    for backend, allocation, median, notes in (
        (
            "tenferro-cuda-to-contiguous",
            True,
            8.0,
            "fresh destination per call",
        ),
        (
            reuse,
            False,
            5.0,
            "caller-owned destination allocated once and reused",
        ),
    )
]
path = tmp / "records.jsonl"
path.write_text("".join(json.dumps(record) + "\n" for record in records))
PY

"${PYTHON[@]}" "$ROOT/scripts/format_gpu_permutation_results.py" \
    "$TMP/records.jsonl" --output "$TMP/report.md"

grep -q 'tenferro-rs CUDA copy_read_into (ms)' "$TMP/report.md"
grep -q 'caller-owned destination' "$TMP/report.md"

"${PYTHON[@]}" - "$ROOT" "$ROOT/src/bin/benchmark_gpu_permutation.rs" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
source = Path(sys.argv[2]).read_text()
arm = source.split('"tenferro-cuda-destination-reuse" =>', 1)[1].split(
    '"cutensor" =>', 1
)[0]
assert "run_tenferro_cuda_destination_reuse" in arm
assert "run_cutensor_permutation" not in arm

implementation = source.split("fn run_tenferro_cuda_destination_reuse", 1)[1].split(
    "fn run_cutensor(", 1
)[0]
assert ".copy_read_into(" in implementation
assert "TensorRead::from_tensor" in implementation
assert "TensorWrite::from_view" in implementation
assert "CutensorLib" not in implementation
assert "run_cutensor" not in implementation
assert "cuda_ptr" not in implementation
assert "per-call destination-view metadata construction included" in implementation

formatter = (root / "scripts" / "format_gpu_permutation_results.py").read_text()
formatter_words = " ".join(formatter.split())
assert (
    "tenferro-cuda-transpose, tenferro-cuda-to-contiguous, "
    "tenferro-cuda-destination-reuse, cutensor, pytorch-cuda"
) in formatter_words

runner = (root / "scripts" / "run_gpu_permutation.sh").read_text()
runner_header = "\n".join(runner.splitlines()[:10])
assert "tenferro-cuda-destination-reuse" in runner_header
status_line = next(
    line for line in runner.splitlines() if line.startswith('echo "Running Rust GPU')
)
assert "tenferro-cuda-destination-reuse" in status_line

docs = (root / "docs" / "gpu-permutation-suite.md").read_text()
assert "per-call destination-view metadata construction" in docs
assert "included in the timed public API dispatch" in docs
PY

echo "GPU permutation destination-reuse contract passed"
