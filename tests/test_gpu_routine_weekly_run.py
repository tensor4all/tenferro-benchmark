"""The unfiltered weekly routine run, end to end, with fake GPU runners.

Nothing here needs a device: the runners are replaced by scripts that emit the
result records the real runners emit, so the test still exercises the parts that
decide what the weekly job collects and publishes — the routine selection, both
suite drivers, the per-suite metadata, the accounting, the report copies and the
bundle's expected rows.
"""
import json
import os
import shlex
import shutil
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from scripts.ci.check_gpu_run import bundle_run, planned_rows, routine_suite_files

TIMESTAMP = "20000101_000000"

FAKE_RUNNER = '''
import json, sys
from pathlib import Path
import yaml

out = Path(sys.argv[1])
problem_filter = sys.argv[3]
sep = sys.argv.index("--")
backends = sys.argv[4:sep] or ["tenferro-cuda-trace"]
rows = []
for suite_path in [Path(p) for p in sys.argv[sep + 1:]]:
    suite = yaml.safe_load(suite_path.read_text())
    for problem in suite["problems"]:
        if problem_filter and problem["id"] != problem_filter:
            continue
        for backend in backends:
            rows.append({
                "schema_version": 1, "suite_id": suite["suite_id"],
                "problem_id": problem["id"], "op": problem.get("op", "matmul"),
                "backend": backend, "status": "ok",
                "timing": {"warmup_runs": 1, "timed_runs": 3, "compile_time_ms": None,
                           "first_run_ms": 1.0, "median_ms": 1.0, "min_ms": 1.0,
                           "p95_ms": 1.0, "iqr_ms": 0.0,
                           "timing_scope": "steady_state_host_api_plus_device_sync"},
                "verification": {"status": "passed", "reference_backend": "cpu_fp64",
                                 "rtol": 1e-08, "atol": 1e-10, "max_abs_error": 1e-12,
                                 "max_rel_error": 1e-13, "residual": 1e-12},
                "execution": {"device": "cuda", "device_ordinal": 0, "execution_path": "fake",
                              "synchronization": "sync", "layout": "{}", "dtype": "{}",
                              "notes": json.dumps({"sampling": {"operations_per_sample": 1,
                                                                "scope": "steady_state_host_api_plus_device_sync",
                                                                "synchronization_policy": "after_each_operation",
                                                                "batch_ms": [1.0, 1.1, 0.9]},
                                                   "setup": "untimed", "precision": "faithful"})}})
out.write_text("".join(json.dumps(row) + "\\n" for row in rows))
'''

FAKE_METADATA = '''
import sys, yaml
from pathlib import Path

args = sys.argv[1:]
value = lambda name: args[args.index(name) + 1]
Path(value("--output")).write_text(yaml.safe_dump({
    "schema_version": 1, "target_profile": value("--target-profile"),
    "suite_id": value("--suite-id"), "suite_file": value("--suite-file"),
    "timestamp": value("--timestamp"),
    "tenferro_rs": {"path": "extern/tenferro-rs", "commit": "a" * 40, "dirty": False,
                    "features": ["cuda"]},
    "harness": {"path": ".", "commit": "b" * 40, "dirty": False},
    "cuda": {"device_name": "NVIDIA A100 80GB PCIe", "device_uuid": "GPU-fake",
             "device_ordinal": 0, "driver_version": "580.0", "runtime_version": "12.8"},
    "environment": {}, "python_backends": {},
}))
'''

FAKE_VERIFY_AD = '''
import shutil, sys
shutil.copyfile(sys.argv[1], sys.argv[2])
'''

# Stands in for scripts/benchmark_gpu_public_api.py: the family shape of its rows
# (case_id, batched per-operation samples, error over tolerance), which the real
# normalizer folds into the routine contract.
FAKE_PUBLIC_API = '''
import json, sys
from pathlib import Path

args = sys.argv[1:]
value = lambda name: args[args.index(name) + 1]
suite_path = value("--suite") if "--suite" in args else None
fixtures = Path(value("--fixtures"))
backend = value("--backend")
if backend == "prepare":
    fixtures.mkdir(parents=True, exist_ok=True)
    (fixtures / "manifest.json").write_text(Path(suite_path).read_text())
    raise SystemExit(0)
# The Rust arm is driven by the prepared fixture manifest, like the real one.
manifest = fixtures if fixtures.suffix == ".json" else fixtures / "manifest.json"
suite = json.loads(Path(suite_path).read_text() if suite_path else manifest.read_text())
rows = []
for case in suite["cases"]:
    rows.append({"schema_version": 1, "suite_id": suite["suite_id"], "case_id": case["id"],
                 "op": case["op"], "backend": backend, "status": "ok",
                 "traffic": {"kind": "logical_traffic", "bytes_per_operation": 536870912,
                             "reads_bytes": 268435456, "writes_bytes": 268435456,
                             "destination_read_included": False, "basis": "fake manifest traffic",
                             "unknown_reason": None},
                 "timing": {"operations_per_sample": 4, "batch_ms": [0.4, 0.5, 0.45], "warmups": 1},
                 "verification": {"max_error_over_tolerance": 1e-3, "reference": "numpy/scipy"}})
Path(value("--output")).write_text("".join(json.dumps(row) + "\\n" for row in rows))
'''

# Stands in for the permutation runners: one record per (pattern, backend) in the
# shape scripts/format_gpu_permutation_results.py and the normalizer read.
FAKE_PERMUTATION = '''
import json, os, sys
from pathlib import Path

suite = json.loads(Path("data/instances/gpu_permutation_patterns.json").read_text())
out = Path(os.environ["BENCH_OUTPUT"])
# The Rust runner takes no backend argument; the Python runner is given one.
backends = sys.argv[1:] or ["tenferro-cuda-transpose", "tenferro-cuda-to-contiguous",
                            "tenferro-cuda-destination-reuse", "cutensor"]
rows = []
for pattern in suite["patterns"]:
    # The real drivers emit a row only for the arms a pattern declares as
    # participants, so the fake gates on the same declaration. Without it the
    # fake invents rows the collector can never produce and hides #368.
    participants = set(pattern.get("participants_gpu") or [])
    for backend in backends:
        if participants and backend not in participants:
            continue
        rows.append({"schema_version": 1, "suite_id": "gpu/permutation", "runner": "fake",
                     "pattern_id": pattern["id"], "label": pattern.get("label", ""),
                     "backend": backend, "shape": pattern["shape"], "perm": pattern["perm"],
                     "dtype": "f64", "elems": 1, "bytes_rw": 16, "device": "cuda:0",
                     "status": "ok", "correctness": "passed", "per_call_allocation": False,
                     "warmup": 1, "iters": 3, "median_ms": 0.01, "p25_ms": 0.009,
                     "p75_ms": 0.011, "bandwidth_gbs": 1.6, "notes": None})
out.write_text("".join(json.dumps(row) + "\\n" for row in rows))
'''

# Stands in for scripts/gpu_copy_calibration.py: the harness has no device, so
# the run metadata gets the same block a real device copy would record.
FAKE_COPY_CALIBRATION = '''
import sys
from pathlib import Path
import yaml

args = sys.argv[1:]
path = Path(args[args.index("--run-metadata") + 1])
metadata = yaml.safe_load(path.read_text())
metadata["gpu_memory_calibration"] = {
    "kind": "device_copy", "method": "fake device copy for a harness without a device",
    "device_ordinal": 0, "device_name": "fake GPU", "buffer_bytes": 67108864,
    "traffic_bytes_per_copy": 134217728, "traffic_model": "one read plus one write of buffer_bytes per copy",
    "warmups": 3, "sample_ms": [0.096], "median_ms": 0.096,
    "bytes_per_second": 1398101333333.3333, "torch_version": "fake",
    "cuda_version": "fake", "timestamp": "2000-01-01T00:00:00+00:00"}
path.write_text(yaml.safe_dump(metadata, sort_keys=False))
'''

# Stands in for `cargo build --bin NAME`: the compiled artifact is a launcher for
# the same fake runner, so the driver code under test is unchanged.
FAKE_CARGO = '''
import sys
from pathlib import Path

name = sys.argv[sys.argv.index("--bin") + 1]
body_path = Path(__file__).with_name("fake_" + name + "_body.py")
if not body_path.is_file():
    body_path = Path(__file__).with_name("fake_runner_body.py")
body = body_path.read_text()
target = Path("target/release")
target.mkdir(parents=True, exist_ok=True)
(target / (name + ".py")).write_text(body)
launcher = target / name
launcher.write_text('#!/bin/sh\\nexec ' + __PYTHON__ + ' "$(dirname "$0")/' + name + '.py" "$@"\\n')
launcher.chmod(0o755)
'''


class WeeklyRoutineRunTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / "scripts", self.root / "scripts",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "schemas", self.root / "schemas")
        shutil.copytree(ROOT / "benchmarks/gpu", self.root / "benchmarks/gpu")
        # Instance documents are what the fixture-driven suites select cases from.
        shutil.copytree(ROOT / "data/instances", self.root / "data/instances")
        (self.root / "scripts/benchmark_gpu_python.py").write_text(FAKE_RUNNER)
        (self.root / "scripts/benchmark_gpu_linalg_ad.py").write_text(FAKE_RUNNER)
        (self.root / "scripts/benchmark_gpu_public_api.py").write_text(FAKE_PUBLIC_API)
        (self.root / "scripts/benchmark_gpu_permutation_python.py").write_text(FAKE_PERMUTATION)
        (self.root / "scripts/verify_gpu_linalg_ad_rust.py").write_text(FAKE_VERIFY_AD)
        (self.root / "scripts/collect_run_metadata.py").write_text(FAKE_METADATA)
        (self.root / "scripts/gpu_copy_calibration.py").write_text(FAKE_COPY_CALIBRATION)
        # Dependency checkout is not part of what this harness exercises.
        (self.root / "scripts/setup_extern_deps.sh").write_text("#!/bin/sh\nexit 0\n")
        # The collectors check for a CUDA toolkit instead of a container; this
        # harness has no device at all, so it provides a stand-in toolkit root.
        self.cuda_home = self.root / "fake-cuda"
        (self.cuda_home / "bin").mkdir(parents=True)
        nvcc = self.cuda_home / "bin" / "nvcc"
        nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 12.4, V12.4.131'\n")
        nvcc.chmod(0o755)
        self.commands = self.root / "bin"
        self.commands.mkdir()
        self.write_command("python", f'exec {shlex.quote(sys.executable)} "$@"')
        self.write_command("python3", f'exec {shlex.quote(sys.executable)} "$@"')
        self.write_command("uv", 'shift; exec "$@"')
        self.write_command("nvidia-smi", "echo GPU-fake")
        self.write_command("cargo", f'exec {shlex.quote(sys.executable)} '
                                   f'"{self.root}/scripts/fake_cargo.py" "$@"')
        (self.root / "scripts/fake_cargo.py").write_text(
            FAKE_CARGO.replace("__PYTHON__", repr(sys.executable)))
        (self.root / "scripts/fake_runner_body.py").write_text(FAKE_RUNNER)
        (self.root / "scripts/fake_benchmark_gpu_permutation_body.py").write_text(FAKE_PERMUTATION)
        (self.root / "scripts/fake_benchmark_gpu_public_api_body.py").write_text(FAKE_PUBLIC_API)

    def write_command(self, name, body):
        script = self.commands / name
        script.write_text("#!/bin/sh\n" + body + "\n")
        script.chmod(0o755)

    def run_weekly(self):
        env = dict(os.environ, PATH=f"{self.commands}:{os.environ['PATH']}",
                   CUDA_HOME=str(self.cuda_home), GPU_BENCH_DEVICE="0", GPU_BENCH_TIMESTAMP=TIMESTAMP,
                   GPU_BENCH_BACKEND_SLEEP="0", GPU_BENCH_ROUTINE_CONTROLS="0",
                   BENCHMARK_TARGET_PROFILE="nvidia-gpu")
        env.pop("GPU_BENCH_SUITE", None)
        env.pop("GPU_BENCH_BACKENDS", None)
        env.pop("GPU_BENCH_PROBLEM", None)
        env.pop("BENCH_COVERAGE", None)
        env.pop("GPU_BENCH_VERIFY_ONLY", None)
        return subprocess.run(["bash", str(self.root / "scripts/run_gpu_suite.sh")],
                              env=env, capture_output=True, text=True, timeout=300)

    def test_unfiltered_run_collects_reports_and_bundles_the_single_set(self):
        process = self.run_weekly()
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        selection = self.root / f"data/results/nvidia-gpu/gpu/routine_selection/{TIMESTAMP}"
        plan = json.loads((selection / "selection.json").read_text())
        self.assertEqual(len(plan["selected"]), 698)
        self.assertNotIn("coverage", plan)
        status = json.loads((selection / "case_status.json").read_text())
        self.assertTrue(status["complete"], status)
        self.assertEqual(len(status["executed"]), 698)
        self.assertEqual(status["counts"]["unexpected"], 0)

        # Every suite recorded its own wall clock, which is what fills the
        # per-suite table of the routine workflow report on any device, and every
        # suite — including the delegated AD ones — shares one campaign
        # timestamp, which the publication gate compares with the run directory.
        timestamps = set()
        for suite_file in routine_suite_files():
            suite_id = (self.root / suite_file).read_text().split("suite_id: ")[1].split("\n")[0]
            run = self.root / f"data/results/nvidia-gpu/gpu/{suite_id.split('/')[1]}/{TIMESTAMP}"
            metadata = json.loads(json.dumps(__import__("yaml").safe_load((run / "run.yaml").read_text())))
            timestamps.add(metadata["timestamp"])
            self.assertGreater(metadata["collection"]["suite_seconds"], 0, suite_id)
            self.assertIn("suite_started_at", metadata["collection"], suite_id)
            self.assertEqual(len(planned_rows(run, suite_id)), len(metadata["collection"]["expected_case_ids"]), suite_id)
            self.assertTrue((run / "report.md").is_file(), suite_id)
            self.assertTrue((run / "report.summary.json").is_file(), suite_id)

        self.assertEqual(len(timestamps), 1, timestamps)

        # A public API row whose logical bytes per operation imply a rate above the
        # run's measured copy rate is published with its number and its ratio, not
        # withheld: the fake rows declare 512 MiB per operation against ~1.4 TB/s.
        public_api_report = (self.root / f"data/results/nvidia-gpu/gpu/structural/{TIMESTAMP}/report.md").read_text()
        self.assertIn("| 112.500 (", public_api_report)
        self.assertIn("| 536870912 | 3.413 |", public_api_report)
        self.assertNotIn("unmeasurable", public_api_report)

        coverage_report = self.root / "result/nvidia-gpu/gpu/routine_coverage.md"
        workflow_report = self.root / "result/nvidia-gpu/gpu/routine_workflow.md"
        self.assertTrue(coverage_report.is_file())
        self.assertTrue(workflow_report.is_file())
        text = coverage_report.read_text()
        self.assertNotIn("coverage:", text)
        self.assertIn("Measured coverage complete: yes.", text)
        self.assertIn("## Per-suite measured time", workflow_report.read_text())

        # The publication gate accepts the curated set: the planned rows are the
        # expected ones, not every case the suite files still contain.
        bundle = self.root / "bundle"
        bundle_run(self.root, TIMESTAMP, [Path(p) for p in routine_suite_files()],
                   ["tenferro-cuda-trace", "tenferro-cuda-eager", "pytorch-cuda", "jax-cuda"],
                   bundle)
        publication = json.loads((bundle / "publication.json").read_text())
        self.assertEqual({entry["suite_id"]: len(entry["expected_rows"]) for entry in publication["suites"]},
                         {"gpu/dense": 28, "gpu/einsum": 4, "gpu/sparse": 2, "gpu/tensornetwork": 4,
                          "gpu/elementwise": 8, "gpu/linalg_ad_latency": 30, "gpu/linalg_jvp_vjp": 30,
                          "gpu/permutation": 44, "gpu/lstsq": 4, "gpu/linalg_public_api_throughput": 64,
                          "gpu/fft": 16, "gpu/nn": 44, "gpu/structural": 64, "gpu/isolated_ops": 132,
                          "gpu/reuse_contraction": 224})

        # The real ingestion path also runs: one campaign timestamp, one
        # tenferro revision and one summary per suite, including the AD lane.
        packaged = subprocess.run(
            [sys.executable, str(self.root / "scripts/ci/prepare_result_bundle.py"),
             "--profile", "nvidia-gpu", "--harness-commit", "b" * 40,
             "--started-at", "2000-01-01T00:00:00Z", "--timestamp", TIMESTAMP,
             "--backends", "tenferro-cuda-trace,tenferro-cuda-eager,pytorch-cuda,jax-cuda,"
                           "jax-cuda-tree-jit",
             "--suites", "all", "--bundle-dir", str(self.root / "ingestion")],
            cwd=self.root, capture_output=True, text=True, timeout=300)
        self.assertEqual(packaged.returncode, 0, packaged.stdout + packaged.stderr)

    def test_hosted_binaries_are_verified_and_not_rebuilt(self):
        # The weekly pod has no Rust toolchain: it receives both binaries from
        # the hosted build job. A stub cargo proves nothing is rebuilt there.
        self.init_git()
        for binary in ("benchmark_gpu_rust", "benchmark_gpu_linalg_ad",
                       "benchmark_gpu_public_api", "benchmark_gpu_permutation"):
            subprocess.run([sys.executable, str(self.root / "scripts/fake_cargo.py"),
                            "build", "--release", "--bin", binary],
                           cwd=self.root, check=True)
            subprocess.run([sys.executable, str(self.root / "scripts/ci/gpu_artifact.py"),
                            f"target/release/{binary}", "--write"], cwd=self.root, check=True)
        self.write_command("cargo", 'echo "cargo must not run on the benchmark pod" >&2; exit 1')
        env = dict(os.environ, PATH=f"{self.commands}:{os.environ['PATH']}",
                   CUDA_HOME=str(self.cuda_home), GPU_BENCH_DEVICE="0", GPU_BENCH_TIMESTAMP=TIMESTAMP,
                   GPU_BENCH_BACKEND_SLEEP="0", GPU_BENCH_ROUTINE_CONTROLS="0",
                   BENCHMARK_TARGET_PROFILE="nvidia-gpu", GPU_BENCH_PREBUILT_RUST="1")
        for name in ("GPU_BENCH_SUITE", "GPU_BENCH_BACKENDS", "GPU_BENCH_PROBLEM", "GPU_BENCH_VERIFY_ONLY"):
            env.pop(name, None)
        process = subprocess.run(["bash", str(self.root / "scripts/run_gpu_suite.sh")],
                                 env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        status = json.loads((self.root / f"data/results/nvidia-gpu/gpu/routine_selection/{TIMESTAMP}"
                             / "case_status.json").read_text())
        self.assertTrue(status["complete"], status)

    def init_git(self):
        for directory in (".", "extern/tenferro-rs"):
            path = self.root / directory
            path.mkdir(parents=True, exist_ok=True)
            run = lambda *args: subprocess.run(["git", "-C", str(path), *args], check=True,
                                               capture_output=True, text=True)
            run("init", "-q")
            run("-c", "user.name=test", "-c", "user.email=test@example.com",
                "commit", "-q", "--allow-empty", "-m", "fixture")

    def test_vendor_arms_are_an_explicit_opt_in_flag(self):
        env = dict(os.environ, PATH=f"{self.commands}:{os.environ['PATH']}",
                   CUDA_HOME=str(self.cuda_home), GPU_BENCH_DEVICE="0", GPU_BENCH_TIMESTAMP=TIMESTAMP,
                   GPU_BENCH_BACKEND_SLEEP="0", GPU_BENCH_ROUTINE_CONTROLS="0",
                   BENCHMARK_TARGET_PROFILE="nvidia-gpu", BENCH_ROUTINE_VENDOR_ARMS="1",
                   GPU_BENCH_PROBLEM="dense_matmul_f64_3072")
        for name in ("GPU_BENCH_SUITE", "GPU_BENCH_BACKENDS"):
            env.pop(name, None)
        process = subprocess.run(["bash", str(self.root / "scripts/run_gpu_suite.sh")],
                                 env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        plan = json.loads((self.root / f"data/results/nvidia-gpu/gpu/routine_selection/{TIMESTAMP}"
                           / "selection.json").read_text())
        self.assertTrue(plan["vendor_arms"])
        dense = [key for key in plan["selected"] if key.startswith("gpu/dense:")]
        self.assertEqual(len(dense), 7)
        self.assertTrue(any(key.endswith(":cublaslt") for key in dense))

    def test_filtered_probe_leaves_the_routine_reports_alone(self):
        env = dict(os.environ, PATH=f"{self.commands}:{os.environ['PATH']}",
                   CUDA_HOME=str(self.cuda_home), GPU_BENCH_DEVICE="0", GPU_BENCH_TIMESTAMP=TIMESTAMP,
                   GPU_BENCH_BACKEND_SLEEP="0", GPU_BENCH_ROUTINE_CONTROLS="0",
                   BENCHMARK_TARGET_PROFILE="nvidia-gpu",
                   GPU_BENCH_SUITE="benchmarks/gpu/dense.yaml",
                   GPU_BENCH_BACKENDS="pytorch-cuda", GPU_BENCH_PROBLEM="dense_matmul_f64_3072")
        process = subprocess.run(["bash", str(self.root / "scripts/run_gpu_suite.sh")],
                                 env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertFalse((self.root / "result/nvidia-gpu/gpu/routine_coverage.md").exists())
        self.assertFalse((self.root / "result/nvidia-gpu/gpu/routine_workflow.md").exists())
        self.assertFalse((self.root / "result/nvidia-gpu/gpu/routine_full_coverage.md").exists())


if __name__ == "__main__":
    unittest.main()
