"""One failing arm or suite never ends the collection.

`run_gpu_suite.sh` records the failure, keeps every row the other arms wrote,
and reports red only after the accounting and the reports are on disk.
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
STAMP = "20000101_000000"
SUITE_FILE = "benchmarks/gpu/dense.yaml"
PROBLEM = "dense_matmul_f64_3072"
BACKEND = "jax-cuda"
FAILING = "pytorch-cuda"


def row(suite_id, problem, backend):
    """A schema-valid `ok` record the formatter can publish."""
    return {
        "schema_version": 1, "suite_id": suite_id, "problem_id": problem, "op": "matmul",
        "backend": backend, "status": "ok",
        "timing": {"warmup_runs": 1, "timed_runs": 3, "compile_time_ms": None,
                   "first_run_ms": 1.6, "median_ms": 1.5, "min_ms": 1.4, "p95_ms": 1.6,
                   "iqr_ms": 0.05, "timing_scope": "steady_state_host_api_plus_device_sync"},
        "verification": {"status": "passed", "reference_backend": "pytorch-cuda",
                         "rtol": 1e-5, "atol": 1e-6},
        "execution": {"device": "cuda", "device_ordinal": 0, "execution_path": "fixture",
                      "synchronization": "fixture", "layout": "col_major", "dtype": "f64",
                      "notes": json.dumps({"sampling": {"batch_ms": [1.5, 1.5, 1.5],
                                                        "operations_per_sample": 1},
                                           "setup": "fixture", "precision": "f64"}),
                      "unsupported_reason": None},
    }


class SuiteContinuesAfterFailureTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for name in ("scripts", "schemas"):
            shutil.copytree(ROOT / name, self.root / name,
                            ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "data/instances", self.root / "data/instances")
        shutil.copytree(ROOT / "benchmarks/gpu/manifests", self.root / "benchmarks/gpu/manifests")
        shutil.copy(ROOT / "benchmarks/gpu/routine_budget.yaml",
                    self.root / "benchmarks/gpu/routine_budget.yaml")
        (self.root / "benchmarks/gpu/dense.yaml").parent.mkdir(parents=True, exist_ok=True)
        import yaml
        manifest = yaml.safe_load((ROOT / "benchmarks/gpu/manifests/routine.yaml").read_text())
        for spec in manifest["suites"].values():
            target = self.root / spec["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / spec["file"], target)
        # A runner that dies for one arm and measures the other. The failed arm
        # writes nothing, exactly like a crashed backend process.
        record = json.dumps(row("gpu/dense", PROBLEM, BACKEND)) + "\n"
        (self.root / "scripts/benchmark_gpu_python.py").write_text(
            "import sys\nfrom pathlib import Path\n"
            f"backend = sys.argv[4]\n"
            f"Path(sys.argv[1]).write_text('' if backend == {FAILING!r} else {record!r})\n"
            f"sys.exit(3 if backend == {FAILING!r} else 0)\n")
        metadata = dict(schema_version=1, target_profile="nvidia-gpu", suite_id="gpu/dense",
                        suite_file=SUITE_FILE, timestamp="2000-01-01T00:00:00Z",
                        tenferro_rs=dict(path="extern/tenferro-rs", commit="abcdef0", dirty=False,
                                         features=["cuda"]), environment={})
        (self.root / "scripts/collect_run_metadata.py").write_text(
            "import sys,yaml\nfrom pathlib import Path\n"
            "Path(sys.argv[sys.argv.index('--output')+1]).write_text(yaml.safe_dump("
            + repr(metadata) + "))\n")
        commands = self.root / "bin"
        commands.mkdir()
        wrappers = {name: f'exec {shlex.quote(sys.executable)} "$@"'
                    for name in ("python", "python3")}
        wrappers.update({"uv": 'shift; exec "$@"', "nvidia-smi": "echo GPU-contract-test"})
        for name, body in wrappers.items():
            script = commands / name
            script.write_text("#!/bin/sh\n" + body + "\n")
            script.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{commands}:{os.environ['PATH']}",
                        GPU_BENCH_TIMESTAMP=STAMP, GPU_BENCH_DEVICE="0",
                        GPU_BENCH_SUITE=SUITE_FILE, GPU_BENCH_BACKENDS=f"{FAILING},{BACKEND}",
                        GPU_BENCH_PROBLEM=PROBLEM, GPU_BENCH_BACKEND_SLEEP="0",
                        GPU_BENCH_SKIP_LATEST="1", GPU_BENCH_ROUTINE_CONTROLS="0")
        for name in ("GPU_BENCH_VERIFY_ONLY",):
            self.env.pop(name, None)
        self.selection = self.root / f"data/results/nvidia-gpu/gpu/routine_selection/{STAMP}"
        self.run_dir = self.root / f"data/results/nvidia-gpu/gpu/dense/{STAMP}"

    def test_a_dead_arm_keeps_the_other_arms_rows_and_still_accounts(self):
        process = subprocess.run(["bash", str(self.root / "scripts/run_gpu_suite.sh")],
                                 env=self.env, capture_output=True, text=True, timeout=120)
        self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
        self.assertIn(f"ARM FAILED: gpu/dense {FAILING} exited 3", process.stderr)
        measured = [json.loads(line) for line in
                    (self.run_dir / "records.jsonl").read_text().splitlines() if line.strip()]
        self.assertEqual([(row["problem_id"], row["backend"]) for row in measured],
                         [(PROBLEM, BACKEND)])
        summary = json.loads((self.run_dir / "report.summary.json").read_text())
        rows = {(row["case"], row["backend"]): row for row in summary["rows"]}
        self.assertEqual(rows[(PROBLEM, BACKEND)]["median"], 1.5)
        unreached = rows[(PROBLEM, FAILING)]
        self.assertEqual(unreached["status"], "missing")
        self.assertIn("no measurement row was written", unreached["reason"])
        # The report on disk names the case it could not measure.
        self.assertIn("## Not measured", (self.run_dir / "report.md").read_text())
        accounting = json.loads((self.selection / "case_status.json").read_text())
        self.assertEqual(accounting["counts"]["executed"], 1)
        self.assertEqual(accounting["counts"]["missing"], 1)
        missing_key = accounting["missing"][0]
        self.assertIn(FAILING, missing_key)
        self.assertIn("no measurement row was written", accounting["reasons"][missing_key])
        self.assertEqual(accounting["statuses"][missing_key], "missing")
        # The coverage report was written even though the run is red.
        self.assertTrue((self.selection / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
