#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

uv run python - <<'PY'
import json
import subprocess
import tempfile
from pathlib import Path

root = Path.cwd()
errors: list[str] = []

for path in sorted((root / "data/instances").glob("*.json")):
    with path.open() as fh:
        instance = json.load(fh)
    intent = instance.get("intent")
    if intent is not None and (not isinstance(intent, str) or not intent.strip()):
        errors.append(f"{path}: optional intent must be a non-empty string")
    notes = instance.get("notes")
    if notes is not None and (not isinstance(notes, str) or not notes.strip()):
        errors.append(f"{path}: optional notes must be a non-empty string")

nary_path = root / "data/instances/nary_matmul_chain_64.json"
if not nary_path.exists():
    errors.append(f"{nary_path}: missing N-ary cache diagnostic instance")
else:
    with nary_path.open() as fh:
        nary = json.load(fh)
    if nary.get("num_tensors") != 3:
        errors.append(f"{nary_path}: expected num_tensors=3")
    if "inner ExecProgram" not in nary.get("intent", ""):
        errors.append(f"{nary_path}: intent should document inner ExecProgram cache coverage")
    if len(nary.get("paths", {}).get("opt_flops", {}).get("path", [])) != 2:
        errors.append(f"{nary_path}: expected two contraction steps")

suite = """\
schema_version: 1
suite_id: cpu/einsum-intent-smoke
title: Intent smoke
description: Verifies optional benchmark problem intent and notes metadata.
defaults:
  run:
    warmups: 1
    runs: 1
backends:
  - tenferro-rs
problems:
  - id: nary_matmul_chain_64
    family: einsum
    op: einsum
    intent: Small N-ary cache diagnostic.
    notes: Smoke-test note for optional benchmark comments.
    dtype:
      values: f64
    data:
      generator: zeros
      seed: 0
    einsum:
      format_rowmajor: ij,jk,kl->il
      format_colmajor: ji,kj,lk->li
      shapes_rowmajor:
        - [64, 64]
        - [64, 64]
        - [64, 64]
      shapes_colmajor:
        - [64, 64]
        - [64, 64]
        - [64, 64]
      path:
        strategy: manual
        pairs:
          - [0, 1]
          - [0, 1]
"""
with tempfile.TemporaryDirectory() as tmp:
    suite_path = Path(tmp) / "suite.yaml"
    suite_path.write_text(suite)
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/validate_benchmark_suite.py",
            "--kind",
            "suite",
            str(suite_path),
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        errors.append(
            "suite schema should allow optional problem intent and notes metadata:\n"
            + result.stderr.strip()
        )

if errors:
    raise SystemExit("\n".join(errors))
PY
