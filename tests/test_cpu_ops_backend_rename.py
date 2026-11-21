#!/usr/bin/env python3
"""The CPU-ops CSV must label the gate's provider rows like render_run expects.

Regression for #507: the publication gate labels its CPU rows ``faer``, but
``run_cpu_ops.sh`` only renamed ``native``/``blas-*``/``cuda`` to
``tenferro-eager``.  ``render_run`` then found raw samples with no CSV row,
never wrote ``run.yaml`` and the nightly bundler reported
``cpu/einsum: missing raw data or metadata``.
"""

import csv
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_run

GATE_FIELDS = ["suite", "op", "phase", "dtype", "backend", "profile", "shape",
               "warmups", "runs", "median_ms", "iqr_ms", "status"]


def rename_script() -> str:
    source = (ROOT / "scripts/run_cpu_ops.sh").read_text()
    match = re.search(r"python3 - .*? <<'PY'\n(.*?)\nPY\n", source, re.DOTALL)
    assert match, "run_cpu_ops.sh no longer embeds the CSV normaliser"
    return match.group(1)


class CpuOpsBackendRename(unittest.TestCase):
    def test_every_gate_provider_label_is_renamed_and_verifies(self):
        for provider in ("native", "blas-openblas", "blas-accelerate", "blas-mkl", "faer", "cuda"):
            with self.subTest(provider=provider), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                gate = run / "gate.csv"
                with gate.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=GATE_FIELDS, lineterminator="\n")
                    writer.writeheader()
                    writer.writerow(dict(suite="small", op="matmul", phase="primal", dtype="f64",
                                         backend=provider, profile="quick", shape="32x32",
                                         warmups=3, runs=3, median_ms="0.001000",
                                         iqr_ms="0.000000", status="ok"))
                (run / "cpu_ops_samples_t1_x.jsonl").write_text(json.dumps({
                    "record_type": "samples", "suite": "small", "benchmark": "matmul",
                    "shape": "32x32", "backend": provider, "threads": 1,
                    "operations_per_sample": 1,
                    "batch_elapsed_ns": [1000, 1000, 1000],
                    "per_operation_ns": [1000.0, 1000.0, 1000.0]}) + "\n")
                out = run / "cpu_ops.csv"
                subprocess.run([sys.executable, "-", str(gate), str(out), "1"],
                               input=rename_script(), text=True, check=True)
                with out.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(["tenferro-eager"], [row["backend"] for row in rows])
                render_run.verify_cpu_ops_samples(run, [out])


if __name__ == "__main__":
    unittest.main()
