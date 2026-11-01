import json
import os
import shlex
import shutil
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import gpu_routine_selection as selection

MANIFEST = ROOT/"benchmarks/gpu/manifests/routine.yaml"


def manifest_suites():
    return [spec["file"] for spec in yaml.safe_load(MANIFEST.read_text())["suites"].values()]


class RoutineSelectionTest(unittest.TestCase):
    def plan(self, output, **kwargs):
        return selection.materialize(MANIFEST, manifest_suites(), output, **kwargs)

    def test_one_set_selects_every_routine_suite_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(Path(tmp))
            self.assertEqual(len(plan["selected"]), 106)
            self.assertEqual(plan["expected"], plan["selected"])
            for suite in ("dense", "einsum", "sparse", "tensornetwork", "elementwise",
                          "linalg_ad_latency", "linalg_jvp_vjp"):
                self.assertTrue(any(key.startswith(f"gpu/{suite}:") for key in plan["selected"]), suite)
            # The sparse suite is reference-only, so it runs one arm.
            self.assertTrue(all(key.endswith(":pytorch-cuda") for key in plan["selected"]
                                if key.startswith("gpu/sparse:")))
            # Size sweeps are not routine: one size per tiny-size AD suite.
            latency = [key for key in plan["selected"] if key.startswith("gpu/linalg_ad_latency:")]
            self.assertEqual(len(latency), 30)
            self.assertEqual({key.split(":")[1].rsplit("_", 1)[1] for key in latency}, {"4"})
            jvp_vjp = [key for key in plan["selected"] if key.startswith("gpu/linalg_jvp_vjp:")]
            self.assertEqual({key.split(":")[1].rsplit("_", 1)[1] for key in jvp_vjp}, {"256"})
            # The diagnostic and vendor arms are not in the default set.
            self.assertFalse(any(key.endswith(":jax-cuda-eager") or "cublaslt" in key
                                 or "cutlass" in key for key in plan["selected"]))
            self.assertFalse(plan["vendor_arms"])

    def test_vendor_arms_are_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(Path(tmp), vendor_arms=True)
            self.assertEqual(len(plan["selected"]), 133)
            self.assertTrue(plan["vendor_arms"])
            dense = [key for key in plan["selected"] if key.startswith("gpu/dense:")]
            self.assertEqual(len(dense), 49)
            self.assertTrue(any(key.endswith(":cublaslt") for key in dense))
            self.assertTrue(any(key.startswith("gpu/sparse:") and key.endswith(":ginkgo")
                                for key in plan["selected"]))

    def test_routine_selection_is_the_materialized_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(Path(tmp))
            suite_ids = yaml.safe_load(MANIFEST.read_text())["suites"]
            self.assertEqual({Path(path).name for path in plan["suite_files"]},
                             {suite_id.replace("/", "_") + ".yaml" for suite_id in suite_ids})
            dense = yaml.safe_load((Path(tmp)/"gpu_dense.yaml").read_text())
            self.assertEqual(len(dense["problems"]), 7)
            self.assertEqual(dense["backends"],
                             ["tenferro-cuda-trace", "tenferro-cuda-eager", "pytorch-cuda", "jax-cuda"])
            self.assertEqual(plan["suite_drivers"][(Path(tmp)/"gpu_dense.yaml").as_posix()], "gpu_suite")
            self.assertEqual(plan["suite_drivers"][(Path(tmp)/"gpu_linalg_ad_latency.yaml").as_posix()],
                             "gpu_linalg_ad")

    def test_effort_does_not_change_case_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            standard=self.plan(Path(tmp)/"standard")
            scan=self.plan(Path(tmp)/"scan",effort="scan")
            self.assertEqual(standard["selected"],scan["selected"])
            for path in scan["suite_files"]:
                for problem in yaml.safe_load(Path(path).read_text())["problems"]:
                    self.assertEqual((problem["run"]["warmups"],problem["run"]["runs"]),(1,3))

    def test_case_outside_the_routine_set_is_not_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=self.plan(Path(tmp),backend_override="pytorch-cuda",problem="dense_matmul_f64_3072")
            self.assertEqual(plan["selected"],["gpu/dense:dense_matmul_f64_3072:pytorch-cuda"])
            with self.assertRaisesRegex(ValueError,"no GPU cases"):
                self.plan(Path(tmp)/"bad",problem="dense_solve_f64_4096_rhs16")

    def test_unknown_backend_and_stale_problem_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT/"benchmarks/gpu", root/"benchmarks/gpu")
            path = root/"benchmarks/gpu/manifests/routine.yaml"
            manifest = yaml.safe_load(path.read_text())
            manifest["suites"]["gpu/dense"]["problems"].append("dense_absent_case")
            path.write_text(yaml.safe_dump(manifest, sort_keys=False))
            with self.assertRaisesRegex(ValueError,"stale routine problem manifest"):
                selection.materialize(path, manifest_suites(), root/"out")
            manifest = yaml.safe_load(MANIFEST.read_text())
            manifest["suites"]["gpu/dense"]["backends"].append("not-a-backend")
            path.write_text(yaml.safe_dump(manifest, sort_keys=False))
            with self.assertRaisesRegex(ValueError,"routine backends absent"):
                selection.materialize(path, manifest_suites(), root/"out")

    def test_unsupported_missing_and_unrecorded_samples_never_count_as_covered(self):
        plan=dict(manifest_version=2,effort="scan",expected=["s:a:b","s:c:b","s:d:b"],
                  selected=["s:a:b","s:c:b","s:d:b"])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"records.jsonl"
            path.write_text(json.dumps(dict(suite_id="s",problem_id="a",backend="b",status="unsupported", execution=dict(unsupported_reason="library does not implement this op")))+"\n"+
                json.dumps(dict(suite_id="s",problem_id="c",backend="b",status="ok",
                                verification=dict(status="passed"),execution=dict(notes=None)))+"\n")
            result=selection.account(plan,[path])
            self.assertEqual(result["executed"],[])
            self.assertEqual(result["unsupported"],["s:a:b"])
            self.assertEqual(result["raw_samples_unavailable"],["s:c:b"])
            self.assertEqual(result["missing"],["s:d:b"])
            self.assertFalse(result["complete"])

    def test_records_outside_the_set_are_unexpected(self):
        plan = dict(manifest_version=2, effort="standard", expected=["s:a:b"], selected=["s:a:b"])
        def row(problem):
            return json.dumps(dict(suite_id="s", problem_id=problem, backend="b", status="ok",
                                   verification=dict(status="passed"),
                                   execution=dict(notes=json.dumps({"sampling": {"batch_ms": [1.0]}}))))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text(row("a") + "\n" + row("c") + "\n")
            status = selection.account(plan, [path])
            self.assertEqual(status["unexpected"], ["s:c:b"])
            self.assertEqual(status["executed"], ["s:a:b"])
            self.assertFalse(status["complete"])

    def record(self, status="skipped", reason="no reference implementation"):
        return dict(schema_version=1, suite_id="gpu/tensornetwork", problem_id="a",
                    op="tensor_network_contract", backend="b", status=status,
                    timing=dict(warmup_runs=0, timed_runs=0, compile_time_ms=None,
                                first_run_ms=None, median_ms=None, min_ms=None,
                                p95_ms=None, iqr_ms=None,
                                timing_scope="steady_state_host_api_plus_device_sync"),
                    verification=dict(status="skipped", reference_backend=None, rtol=None, atol=None),
                    execution=dict(device="cuda", device_ordinal=0, execution_path="test",
                                   synchronization="not executed", layout="", dtype="",
                                   unsupported_reason=reason))

    def test_cli_classifies_benign_rows_but_preserves_every_failure(self):
        key = "gpu/tensornetwork:a:b"
        runtime_reason = ("run: TypedTensor::host_data: runtime state failure: backend storage "
                          "cannot be borrowed as host data; download explicitly first")
        scenarios = [
            ([self.record()], "skipped", 0),
            ([self.record("unsupported", "GPU SKU does not support this operation")], "unsupported", 0),
            ([self.record("not_configured", "optional reference library not installed")], "skipped", 0),
            ([self.record("runtime_failed", runtime_reason)], "failed", 1),
            ([self.record("runtime_failed", "case timed out")], "failed", 1),
            ([self.record("compile_failed", "compiler error")], "failed", 1),
            ([self.record("verification_failed", "value mismatch")], "failed", 1),
            ([self.record("skipped", " ")], "failed", 1),
            ([self.record("unsupported", None)], "failed", 1),
            ([], "missing", 1),
            ([self.record(), self.record()], "duplicates", 1),
            ([self.record(), dict(self.record(), problem_id="unknown")], "unexpected", 1),
        ]
        contradictory = self.record("unsupported", "not an excuse for wrong values")
        contradictory["verification"]["status"] = "failed"
        scenarios.append(([contradictory], "failed", 1))
        for rows, category, code in scenarios:
            with self.subTest(category=category, rows=rows), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                plan = dict(manifest_version=2, effort="scan",
                            expected=[key], selected=[key])
                (output/"selection.json").write_text(json.dumps(plan))
                records = output/"records.jsonl"
                records.write_text("".join(json.dumps(row)+"\n" for row in rows))
                process = subprocess.run([sys.executable, str(ROOT/"scripts/gpu_routine_selection.py"),
                    "--output", tmp, "--records", str(records)], capture_output=True, text=True)
                self.assertEqual(process.returncode, code, process.stderr)
                status = json.loads((output/"case_status.json").read_text())
                self.assertEqual(status["counts"][category], 1, status)
                self.assertIn(f"{category}: 1", process.stdout)
                if category in {"skipped", "unsupported", "failed"}:
                    self.assertTrue(status["reasons"][key].strip())
                    self.assertIn(status["reasons"][key], process.stdout)
                self.assertFalse(status["complete"])

    def test_selected_duplicates_and_unknown_ids_still_fail(self):
        key = "gpu/tensornetwork:a:b"
        plan = dict(manifest_version=2, effort="scan", expected=[key], selected=[key])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"records.jsonl"
            rows = [self.record(), self.record(), dict(self.record(), problem_id="unknown")]
            path.write_text("".join(json.dumps(row)+"\n" for row in rows))
            status = selection.account(plan, [path])
            self.assertEqual(status["duplicates"], [key])
            self.assertEqual(status["unexpected"], ["gpu/tensornetwork:unknown:b"])
            self.assertFalse(status["complete"])

    def test_collector_writes_accounting_before_error_exit(self):
        invalid = self.record("skipped", "reference unavailable")
        invalid["unexpected_field"] = True
        scenarios = [(self.record("runtime_failed", "execution exception"), "1", 1, "failed"),
                     (self.record("unsupported", " "), "0", 1, "failed"),
                     (self.record("skipped", "reference unavailable"), "1", 0, "skipped"),
                     (self.record("unsupported", "GPU SKU unsupported"), "1", 0, "unsupported"),
                     (invalid, "0", 1, "skipped")]
        for record, verify_only, code, category in scenarios:
            with self.subTest(record=record), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                shutil.copytree(ROOT/"scripts", root/"scripts", ignore=shutil.ignore_patterns("__pycache__"))
                shutil.copytree(ROOT/"schemas", root/"schemas")
                shutil.copytree(ROOT/"benchmarks/gpu/manifests", root/"benchmarks/gpu/manifests")
                shutil.copy(ROOT/"benchmarks/gpu/routine_budget.yaml", root/"benchmarks/gpu/routine_budget.yaml")
                for suite_file in manifest_suites():
                    shutil.copy(ROOT/suite_file, root/suite_file)
                row = dict(record)
                row.update(suite_id="gpu/dense", problem_id="dense_matmul_f64_3072",
                           backend="pytorch-cuda", op="matmul")
                (root/"scripts/benchmark_gpu_python.py").write_text(
                    "import sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(" +
                    repr(json.dumps(row)+"\n") + ")\n")
                metadata = dict(schema_version=1, target_profile="nvidia-gpu", suite_id="gpu/dense",
                                suite_file="benchmarks/gpu/dense.yaml", timestamp="2000-01-01T00:00:00Z",
                                tenferro_rs=dict(path="extern/tenferro-rs", commit="abcdef0", dirty=False,
                                                 features=["cuda"]), environment={})
                (root/"scripts/collect_run_metadata.py").write_text(
                    "import sys,yaml\nfrom pathlib import Path\n"
                    "Path(sys.argv[sys.argv.index('--output')+1]).write_text(yaml.safe_dump(" +
                    repr(metadata) + "))\n")
                commands = root/"bin"
                commands.mkdir()
                wrappers = {name: f'exec {shlex.quote(sys.executable)} "$@"' for name in ("python", "python3")}
                wrappers.update({"uv": 'shift; exec "$@"', "nvidia-smi": "echo GPU-contract-test"})
                for name, body in wrappers.items():
                    script = commands/name
                    script.write_text("#!/bin/sh\n"+body+"\n")
                    script.chmod(0o755)
                env = dict(os.environ, PATH=f"{commands}:{os.environ['PATH']}",
                           GPU_BENCH_VERIFY_ONLY=verify_only, GPU_BENCH_DEVICE="0",
                           GPU_BENCH_TIMESTAMP="20000101_000000",
                           GPU_BENCH_SUITE="benchmarks/gpu/dense.yaml",
                           GPU_BENCH_BACKENDS="pytorch-cuda", GPU_BENCH_PROBLEM="dense_matmul_f64_3072",
                           GPU_BENCH_BACKEND_SLEEP="0", GPU_BENCH_SKIP_LATEST="1",
                           GPU_BENCH_ROUTINE_CONTROLS="0")
                process = subprocess.run(["bash", str(root/"scripts/run_gpu_suite.sh")],
                                         env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, code, process.stdout+process.stderr)
                kind = "verification" if verify_only == "1" else "results"
                output = root/f"data/{kind}/nvidia-gpu/gpu/routine_selection/20000101_000000/case_status.json"
                self.assertTrue(output.is_file(), process.stdout+process.stderr)
                accounting = json.loads(output.read_text())
                self.assertEqual(accounting["counts"][category], 1)
                self.assertIn(f"{category}: 1", process.stdout)

    def test_schema_requires_nonblank_reasons_and_rejects_contradictions(self):
        import copy
        import jsonschema
        schema = json.loads((ROOT/"schemas/benchmark-result.schema.json").read_text())
        validator = jsonschema.Draft202012Validator(schema)
        for status in ("skipped", "unsupported", "not_configured"):
            with self.subTest(status=status):
                row = self.record(status)
                validator.validate(row)
                for value in (None, "", " \n\t"):
                    invalid = copy.deepcopy(row)
                    invalid["execution"]["unsupported_reason"] = value
                    self.assertTrue(list(validator.iter_errors(invalid)))
                del row["execution"]["unsupported_reason"]
                self.assertTrue(list(validator.iter_errors(row)))
                row["verification"]["reason"] = "no reference for this operation"
                validator.validate(row)
                row["verification"]["status"] = "failed"
                self.assertTrue(list(validator.iter_errors(row)))


if __name__ == "__main__": unittest.main()
