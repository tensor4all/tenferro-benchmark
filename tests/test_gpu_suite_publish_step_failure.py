"""A failing publish step never withholds the accounting.

`run_gpu_suite.sh` publishes each suite's page, its summary sidecar and the
ingestion receipt inside the collection loop, then writes `case_status.json` and
the coverage report once the loop is done, and exits 1 last. A receipt that
refuses a page therefore has to be recorded and skipped like every other step:
ending the script there would leave the run red *before* the accounting that
says which cases were measured, which is the loss #374 forbids.

The collection runs against a fixture root, so no GPU and no network are used.
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
BACKENDS = ("jax-cuda", "pytorch-cuda")


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


class PublishStepFailureTest(unittest.TestCase):
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
        import yaml
        manifest = yaml.safe_load((ROOT / "benchmarks/gpu/manifests/routine.yaml").read_text())
        for spec in manifest["suites"].values():
            target = self.root / spec["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / spec["file"], target)
        # A runner that measures both arms: this suite is not what fails.
        record = json.dumps(row("gpu/dense", PROBLEM, "placeholder"))
        (self.root / "scripts/benchmark_gpu_python.py").write_text(
            "import json, sys\nfrom pathlib import Path\n"
            f"record = json.loads({record!r})\n"
            "record['backend'] = sys.argv[4]\n"
            "Path(sys.argv[1]).write_text(json.dumps(record) + '\\n')\n"
            "sys.exit(0)\n")
        # The wall report reads `harness` and `cuda`, so the fixture supplies them:
        # without them the run would be red for a reason that has nothing to do
        # with the publish steps, and a missing failure flag would go unnoticed.
        metadata = dict(schema_version=1, target_profile="nvidia-gpu", suite_id="gpu/dense",
                        suite_file=SUITE_FILE, timestamp="2000-01-01T00:00:00Z",
                        tenferro_rs=dict(path="extern/tenferro-rs", commit="abcdef0", dirty=False,
                                         features=["cuda"]),
                        harness=dict(commit="abcdef0", dirty=False),
                        cuda=dict(device_ordinal=0, device_name="GPU-contract-test",
                                  device_uuid="GPU-0000"),
                        environment={})
        (self.root / "scripts/collect_run_metadata.py").write_text(
            "import sys,yaml\nfrom pathlib import Path\n"
            "Path(sys.argv[sys.argv.index('--output')+1]).write_text(yaml.safe_dump("
            + repr(metadata) + "))\n")
        # The ingestion receipt is the publish step under test: `write_receipt`
        # makes it refuse the page, `publish_ok` leaves the real writer in place.
        self.receipt = self.root / "scripts/ci/result_ingestion.py"
        self.real_receipt = self.receipt.read_text()
        self.receipt.write_text(
            "import sys\nprint('receipt: refused ' + sys.argv[1], file=sys.stderr)\nsys.exit(4)\n")
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
                        GPU_BENCH_SUITE=SUITE_FILE, GPU_BENCH_BACKENDS=",".join(BACKENDS),
                        GPU_BENCH_PROBLEM=PROBLEM, GPU_BENCH_BACKEND_SLEEP="0",
                        GPU_BENCH_ROUTINE_CONTROLS="0")
        # The publish path only runs when the maintained page is refreshed.
        for name in ("GPU_BENCH_VERIFY_ONLY", "GPU_BENCH_SKIP_LATEST"):
            self.env.pop(name, None)
        self.selection = self.root / f"data/results/nvidia-gpu/gpu/routine_selection/{STAMP}"
        self.run_dir = self.root / f"data/results/nvidia-gpu/gpu/dense/{STAMP}"

    def run_collection(self):
        return subprocess.run(["bash", str(self.root / "scripts/run_gpu_suite.sh")],
                              env=self.env, capture_output=True, text=True, timeout=120)

    def test_a_refused_page_still_publishes_the_rows_and_the_accounting(self):
        process = self.run_collection()
        # Red, but after everything that measured is written.
        self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
        self.assertIn("receipt: refused", process.stderr)
        # The page and its maintained sidecar were published before the receipt ran.
        self.assertTrue((self.root / "result/nvidia-gpu/gpu/dense.md").is_file())
        maintained = self.root / "result/nvidia-gpu/gpu/dense.summary.json"
        self.assertEqual(maintained.read_text(), (self.run_dir / "report.summary.json").read_text())
        # The accounting, the coverage report and the wall report name what was
        # measured, and the failure is a named one, not a bare `set -e` exit.
        accounting = json.loads((self.selection / "case_status.json").read_text())
        self.assertEqual(accounting["counts"]["executed"], 2)
        self.assertTrue((self.selection / "report.md").is_file())
        self.assertTrue((self.selection / "wall_report.md").is_file())
        self.assertIn("PUBLISH FAILED", process.stderr)
        # The refused receipt wrote no receipt, so nothing attests the page.
        self.assertFalse((self.root / "result/nvidia-gpu/gpu/dense.provenance.json").is_file())

    def test_a_stale_page_is_never_attested(self):
        # A previous run's page and sidecar are already in `result/`. A publish that
        # cannot replace them must not hand them to the receipt: the receipt hashes
        # whatever it finds there, so attesting would publish the old numbers as
        # this run's, which is worse than a missing receipt because it looks done.
        self.receipt.write_text(self.real_receipt)
        first = self.run_collection()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        page = self.root / "result/nvidia-gpu/gpu/dense.md"
        sidecar = self.root / "result/nvidia-gpu/gpu/dense.summary.json"
        receipt = self.root / "result/nvidia-gpu/gpu/dense.provenance.json"
        receipt.unlink()
        page.chmod(0o444)
        sidecar.chmod(0o444)
        self.addCleanup(page.chmod, 0o644)
        self.addCleanup(sidecar.chmod, 0o644)
        before = (page.read_text(), sidecar.read_text())
        second = self.run_collection()
        self.assertEqual(second.returncode, 1, second.stdout + second.stderr)
        self.assertIn("PUBLISH FAILED", second.stderr)
        # The stale pair is untouched and unattested.
        self.assertEqual((page.read_text(), sidecar.read_text()), before)
        self.assertFalse(receipt.exists(), "a receipt attested a page this run could not publish")
        # The accounting still names what this run measured.
        accounting = json.loads((self.selection / "case_status.json").read_text())
        self.assertEqual(accounting["counts"]["executed"], 2)

    def test_a_page_the_receipt_accepts_keeps_the_run_green(self):
        # The control: the same fixture with the real writer is not red, so the
        # assertions above are about the refused receipt and not about the fixture.
        self.receipt.write_text(self.real_receipt)
        process = self.run_collection()
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertNotIn("PUBLISH FAILED", process.stderr)
        self.assertTrue((self.root / "result/nvidia-gpu/gpu/dense.provenance.json").is_file())


if __name__ == "__main__":
    unittest.main()
