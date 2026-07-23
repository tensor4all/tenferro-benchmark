#!/usr/bin/env bash
set -euo pipefail

# Guards against drift between src/bin/benchmark_permutation.rs /
# scripts/benchmark_permutation.jl and schemas/permutation-result.schema.json:
# runs the real Rust runner (and the Julia runner, if `julia` is on PATH) for
# one cheap pattern and validates the actual JSONL output against the schema,
# then checks the schema and the formatter's fail-hard behavior on synthetic
# bad records.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PYTHON=(python3)
if command -v uv >/dev/null 2>&1; then
    PYTHON=(uv run python)
fi

validate_permutation_result() {
    "${PYTHON[@]}" scripts/validate_benchmark_suite.py --kind permutation-result "$1"
}

# --- Pattern participants must match backend capability rules ---------------

"${PYTHON[@]}" - <<'PY'
import json
from pathlib import Path

import yaml

patterns_path = Path("data/instances/permutation_patterns.json")
suite_path = Path("benchmarks/cpu/permutation.yaml")
patterns = json.loads(patterns_path.read_text())["patterns"]
suite_backends = set(yaml.safe_load(suite_path.read_text())["backends"])

for pattern in patterns:
    pattern_id = pattern["id"]
    participants = set(pattern["participants"])
    src_col_major = pattern["src_layout"]["kind"] == "col_major"
    dst_col_major = pattern["dst_layout"]["kind"] == "col_major"
    identity = pattern["perm"] == list(range(len(pattern["perm"])))

    unknown = participants - suite_backends
    assert not unknown, f"{pattern_id}: unknown CPU participants: {sorted(unknown)}"

    if "hptt" in participants:
        assert src_col_major and dst_col_major, (
            f"{pattern_id}: HPTT requires contiguous source and destination"
        )
    if pattern_id == "memcpy_24d_contiguous":
        assert identity and src_col_major and dst_col_major
        assert participants == {"memcpy", "strided-rs"}
        continue

    required = {
        "tenferro-rs",
        "strided-rs",
        "julia-base",
        "strided-jl",
    }
    assert required <= participants, (
        f"{pattern_id}: missing common participants: {sorted(required - participants)}"
    )

    if src_col_major and dst_col_major:
        assert "hptt" in participants, (
            f"{pattern_id}: HPTT-expressible pattern must exercise HPTT"
        )
    else:
        assert "hptt" not in participants
PY

# --- Real Rust runner output must satisfy the schema -----------------------

cargo build --release --bin benchmark_permutation >/dev/null

RUST_JSONL="$TMP/rust_output.jsonl"
PATTERN_ID=transpose_2d_2048 BENCH_RUNS=1 BENCH_WARMUPS=0 \
    BENCH_OUTPUT="$RUST_JSONL" ./target/release/benchmark_permutation >/dev/null
test -s "$RUST_JSONL"
validate_permutation_result "$RUST_JSONL"
grep -q '"backend":"tenferro-rs"' "$RUST_JSONL"
! grep -q '"backend":"naive"' "$RUST_JSONL"
! grep -q '"backend":"tenferro-transpose"' "$RUST_JSONL"
! grep -q '"backend":"tenferro-to-contiguous"' "$RUST_JSONL"
grep -q '"runner":"rust"' "$RUST_JSONL"

# --- Real Julia runner output must satisfy the schema (skip if no julia) ---

if command -v julia >/dev/null 2>&1; then
    JULIA_JSONL="$TMP/julia_output.jsonl"
    (
        cd "$ROOT"
        julia --project="$ROOT" -e 'import Pkg; Pkg.instantiate()' >/dev/null
        PATTERN_ID=transpose_2d_2048 BENCH_RUNS=1 BENCH_WARMUPS=0 BENCH_OUTPUT="$JULIA_JSONL" \
            julia --project="$ROOT" scripts/benchmark_permutation.jl >/dev/null
    )
    test -s "$JULIA_JSONL"
    validate_permutation_result "$JULIA_JSONL"
    grep -q '"backend":"julia-base"' "$JULIA_JSONL"
    grep -q '"runner":"julia"' "$JULIA_JSONL"
else
    echo "julia not found on PATH; skipping Julia runner schema check" >&2
    JULIA_JSONL=""
fi

# --- The formatter must fail hard, not warn, on a schema violation ---------

BAD_JSONL="$TMP/bad_output.jsonl"
cat >"$BAD_JSONL" <<'JSON'
{"schema_version":2,"suite_id":"cpu/permutation","runner":"rust","pattern_id":"transpose_2d_2048","label":"2D 2048^2 transpose [1,0]","backend":"not-a-real-backend","shape":[2048,2048],"perm":[1,0],"dtype":"f64","elems":4194304,"bytes_rw":67108864,"threads":1,"status":"ok","correctness":"passed","per_call_allocation":false,"warmup":0,"iters":1,"median_ms":0.1,"p25_ms":0.1,"p75_ms":0.1,"gbps":1.0,"notes":null}
JSON
if validate_permutation_result "$BAD_JSONL" >/dev/null 2>&1; then
    echo "invalid backend name unexpectedly passed permutation-result validation" >&2
    exit 1
fi
if "${PYTHON[@]}" scripts/format_permutation_results.py "$BAD_JSONL" >/dev/null 2>"$TMP/format_stderr"; then
    echo "format_permutation_results.py unexpectedly accepted a schema-invalid record" >&2
    cat "$TMP/format_stderr" >&2
    exit 1
fi

# --- status/correctness cross-field rule must be enforced ------------------

INCONSISTENT_JSONL="$TMP/inconsistent_output.jsonl"
cat >"$INCONSISTENT_JSONL" <<'JSON'
{"schema_version":2,"suite_id":"cpu/permutation","runner":"rust","pattern_id":"transpose_2d_2048","label":"2D 2048^2 transpose [1,0]","backend":"tenferro-rs","shape":[2048,2048],"perm":[1,0],"dtype":"f64","elems":4194304,"bytes_rw":67108864,"threads":1,"status":"ok","correctness":"failed","per_call_allocation":false,"warmup":0,"iters":1,"median_ms":0.1,"p25_ms":0.1,"p75_ms":0.1,"gbps":1.0,"notes":null}
JSON
if validate_permutation_result "$INCONSISTENT_JSONL" >/dev/null 2>&1; then
    echo "status=ok with correctness=failed unexpectedly passed validation" >&2
    exit 1
fi

# --- successful records must use common allocation-inclusive semantics ------

REUSED_DST_JSONL="$TMP/reused_dst_output.jsonl"
cat >"$REUSED_DST_JSONL" <<'JSON'
{"schema_version":2,"suite_id":"cpu/permutation","runner":"rust","pattern_id":"transpose_2d_2048","label":"2D 2048^2 transpose [1,0]","backend":"tenferro-rs","shape":[2048,2048],"perm":[1,0],"dtype":"f64","elems":4194304,"bytes_rw":67108864,"threads":1,"status":"ok","correctness":"passed","per_call_allocation":false,"warmup":0,"iters":1,"median_ms":0.1,"p25_ms":0.1,"p75_ms":0.1,"gbps":1.0,"notes":null}
JSON
if validate_permutation_result "$REUSED_DST_JSONL" >/dev/null 2>&1; then
    echo "status=ok with per_call_allocation=false unexpectedly passed validation" >&2
    exit 1
fi

# --- suite YAML must still validate -----------------------------------------

"${PYTHON[@]}" scripts/validate_benchmark_suite.py benchmarks/cpu/permutation.yaml

echo "cpu/permutation result schema test passed"
