#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python3 - "$ROOT/src/main.rs" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1]).read_text()


def section(start: str, end: str) -> str:
    try:
        begin = src.index(start)
        finish = src.index(end, begin)
    except ValueError as exc:
        raise SystemExit(f"missing marker: {exc}") from exc
    return src[begin:finish]


trace = section("fn run_instance_trace", "fn contract_once_eager")
try:
    timed_trace = trace.split("// Timed runs", 1)[1].split(
        "if profile_bench_breakdown_enabled()", 1
    )[0]
except IndexError as exc:
    raise SystemExit("run_instance_trace has no timed-run marker") from exc

if "create_operand_tensors" in timed_trace:
    raise SystemExit("run_instance_trace creates input tensors inside timed runs")

eager_contract = section("fn contract_once_eager", "fn run_instance_eager")
if "create_eager_operands" in eager_contract:
    raise SystemExit("contract_once_eager creates input tensors instead of reusing prepared inputs")

for needle in (
    "TENFERRO_PROFILE_BENCH_BREAKDOWN",
    "trace.input_create",
    "trace.executor_run",
    "eager.input_create",
    "eager.einsum_call",
):
    if needle not in src:
        raise SystemExit(f"missing benchmark timing breakdown marker: {needle}")
PY
