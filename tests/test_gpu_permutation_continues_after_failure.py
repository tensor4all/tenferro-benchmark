"""One failing gpu/permutation arm never ends the collection (#374).

`run_gpu_permutation.sh` runs a Rust collector and then one process per Python
arm. A dead arm used to end the script under `set -e`, before the formatter ran:
the rows the other arms had already written were never folded, so the suite
published nothing and the week lost the measurements it did make. The script
records the failure, keeps every row the other arms wrote, writes the report,
the summary and the routine-contract records, and reports red only after that.

The second test covers publication itself; the first one runs with
`GPU_BENCH_SKIP_LATEST=1`, so it says nothing about it.
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
SUITE_ID = "gpu/permutation"
SUITE_FILE = "benchmarks/gpu/permutation.yaml"
PATTERN = "transpose_2d_32768_16384"
RUST_ARM = "tenferro-cuda-transpose"
MEMCPY_ARM = "memcpy-d2d"
FAILING_ARM = "pytorch-cuda"


def row(backend):
    """One permutation record in the shape both runners write."""
    return {
        "schema_version": 2, "suite_id": SUITE_ID, "runner": "fixture",
        "pattern_id": PATTERN, "label": "2D 32768x16384 transpose [1,0]",
        "backend": backend, "shape": [32768, 16384], "perm": [1, 0], "dtype": "f64",
        "elems": 2 ** 29, "bytes_rw": 2 ** 32, "device": "NVIDIA A100 80GB PCIe",
        "status": "ok", "correctness": "passed", "per_call_allocation": True,
        "warmup": 1, "iters": 3, "median_ms": 1.5, "p25_ms": 1.4, "p75_ms": 1.6,
        "gbps": 100.0, "notes": None,
    }


class PermutationContinuesAfterFailureTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for name in ("scripts", "schemas"):
            shutil.copytree(ROOT / name, self.root / name,
                            ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "data/instances", self.root / "data/instances")
        (self.root / SUITE_FILE).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / SUITE_FILE, self.root / SUITE_FILE)
        # The publish step imports the routine manifest for its suite inventory.
        (self.root / "benchmarks/gpu/manifests").mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "benchmarks/gpu/manifests/routine.yaml",
                    self.root / "benchmarks/gpu/manifests/routine.yaml")

        # A Rust collector that measures its one arm and a Python runner whose
        # pytorch arm dies before writing anything, exactly like a crashed
        # backend process. The other Python arm still measures.
        rust = self.root / "target/release/benchmark_gpu_permutation"
        rust.parent.mkdir(parents=True, exist_ok=True)
        rust.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os\n"
            f"open(os.environ['BENCH_OUTPUT'], 'a').write(json.dumps({row(RUST_ARM)!r}) + chr(10))\n")
        rust.chmod(0o755)
        (self.root / "scripts/benchmark_gpu_permutation_python.py").write_text(
            "import json, os, sys\n"
            f"backend = sys.argv[1]\n"
            f"if backend == {FAILING_ARM!r}:\n"
            "    sys.exit(3)\n"
            "open(os.environ['BENCH_OUTPUT'], 'w').write("
            f"json.dumps({row(MEMCPY_ARM)!r}) + chr(10))\n")
        # `GPU_BENCH_PREBUILT_RUST=1` ships the binary instead of building it.
        (self.root / "scripts/ci/gpu_artifact.py").write_text("raise SystemExit(0)\n")

        planned = [f"{SUITE_ID}:{PATTERN}:{backend}" for backend in (RUST_ARM, MEMCPY_ARM, FAILING_ARM)]
        metadata = dict(
            schema_version=1, target_profile="nvidia-gpu", suite_id=SUITE_ID,
            suite_file=SUITE_FILE, timestamp="2000-01-01T00:00:00Z",
            tenferro_rs=dict(path="extern/tenferro-rs", commit="a" * 40, dirty=False,
                             features=["cuda"]),
            cuda=dict(device_name="NVIDIA A100 80GB PCIe"),
            collection=dict(
                effort="standard", gpu_routine_manifest_version=2,
                gpu_routine_selection_file="data/results/nvidia-gpu/gpu/routine_selection/"
                                           f"{STAMP}/selection.json",
                gpu_routine_vendor_arms=False,
                selected_case_ids=planned, expected_case_ids=planned))
        (self.root / "scripts/collect_run_metadata.py").write_text(
            "import sys, yaml\nfrom pathlib import Path\n"
            "Path(sys.argv[sys.argv.index('--output')+1]).write_text(yaml.safe_dump("
            + repr(metadata) + "))\n")

        commands = self.root / "bin"
        commands.mkdir()
        wrappers = {name: f'exec {shlex.quote(sys.executable)} "$@"'
                    for name in ("python", "python3")}
        wrappers["uv"] = 'shift; exec "$@"'
        for name, body in wrappers.items():
            script = commands / name
            script.write_text("#!/bin/sh\n" + body + "\n")
            script.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{commands}:{os.environ['PATH']}",
                        BENCHMARK_TARGET_PROFILE="nvidia-gpu",
                        GPU_BENCH_TIMESTAMP=STAMP, GPU_BENCH_DEVICE="0",
                        GPU_BENCH_SKIP_LATEST="1", GPU_BENCH_BACKEND_SLEEP="0",
                        GPU_BENCH_PREBUILT_RUST="1", GPU_BENCH_NO_PULL="1")
        self.run_dir = self.root / f"data/results/nvidia-gpu/gpu/permutation/{STAMP}"

    def test_a_dead_arm_keeps_the_other_arms_rows_and_still_accounts(self):
        process = subprocess.run(["bash", str(self.root / "scripts/run_gpu_permutation.sh")],
                                 env=self.env, capture_output=True, text=True, timeout=300)
        output = process.stdout + process.stderr
        # The run is red, but only after everything it measured is on disk.
        self.assertEqual(process.returncode, 1, output)
        self.assertIn(f"ARM FAILED: {SUITE_ID} {FAILING_ARM} exited 3", process.stderr)
        summary = json.loads((self.run_dir / "report.summary.json").read_text())
        rows = {(row["case"], row["backend"]): row for row in summary["rows"]}
        self.assertEqual(rows[(PATTERN, RUST_ARM)]["median"], 1.5)
        self.assertEqual(rows[(PATTERN, MEMCPY_ARM)]["median"], 1.5)
        unreached = rows[(PATTERN, FAILING_ARM)]
        self.assertEqual(unreached["status"], "missing")
        self.assertIn("no measurement row was written", unreached["reason"])
        measured = [json.loads(line) for line in
                    (self.run_dir / "records.jsonl").read_text().splitlines() if line.strip()]
        self.assertEqual({(record["problem_id"], record["backend"]) for record in measured},
                         {(PATTERN, RUST_ARM), (PATTERN, MEMCPY_ARM)})
        self.assertTrue((self.run_dir / "report.md").is_file())
    def test_a_page_without_its_summary_is_not_published(self):
        # Publication enabled, and the two steps that produce what a publish
        # needs disagree: the formatter renders the page and dies before writing
        # the sidecar, and the normalizer refuses the same rows. A page whose
        # sidecar and receipt do not exist is half a run, so nothing may reach
        # result/ -- the maintained page, its summary and its receipt publish
        # together or not at all.
        self.env.pop("GPU_BENCH_SKIP_LATEST")
        (self.root / "scripts/format_gpu_permutation_results.py").write_text(
            "import sys\nfrom pathlib import Path\n"
            "Path(sys.argv[sys.argv.index('--output')+1]).write_text('# half a run\\n')\n"
            "sys.exit(7)\n")
        (self.root / "scripts/normalize_gpu_records.py").write_text(
            "import sys\nsys.exit(9)\n")
        process = subprocess.run(["bash", str(self.root / "scripts/run_gpu_permutation.sh")],
                                 env=self.env, capture_output=True, text=True, timeout=300)
        output = process.stdout + process.stderr
        self.assertEqual(process.returncode, 1, output)
        self.assertIn("FORMAT FAILED: gpu/permutation formatter exited 7", process.stderr)
        self.assertIn("NORMALIZE FAILED: gpu/permutation normalize exited 9", process.stderr)
        self.assertIn("PUBLISH SKIPPED", process.stderr)
        self.assertFalse((self.root / "result/nvidia-gpu/gpu/permutation.md").exists(), output)
        self.assertFalse((self.root / "result/nvidia-gpu/gpu/permutation.summary.json").exists(),
                         output)


if __name__ == "__main__":
    unittest.main()
