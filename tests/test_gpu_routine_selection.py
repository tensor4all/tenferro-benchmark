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


class RoutineSelectionTest(unittest.TestCase):
    def plan(self, output, coverage="quick", **kwargs):
        return selection.materialize(ROOT/"benchmarks/gpu/manifests/routine.yaml",
            ["benchmarks/gpu/"+name+".yaml" for name in ("dense","einsum","sparse","tensornetwork")],
            output,coverage,**kwargs)

    def test_quick_retains_core_mechanisms_and_full_preserves_all_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            quick=self.plan(Path(tmp)/"quick"); full=self.plan(Path(tmp)/"full","full")
            self.assertEqual(len(quick["selected"]),27)
            self.assertEqual(len(full["selected"]),94)
            self.assertTrue(set(quick["selected"])<=set(full["selected"]))
            self.assertTrue(any("b16_32" in key for key in quick["selected"]))
            self.assertTrue(any("tensornetwork" in key for key in quick["selected"]))
            self.assertFalse(any("gpu/sparse:" in key or "4096" in key for key in quick["selected"]))
            # The diagnostic eager JAX tensor-network path is opt-in, not default full.
            self.assertFalse(any(key.endswith(":jax-cuda-eager") for key in full["selected"]))

    def test_jax_cuda_eager_tensor_network_is_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(
                Path(tmp), "full",
                backend_override="jax-cuda-eager",
                problem="tensornetwork_permutation_optimized_f32",
            )
            self.assertEqual(
                plan["selected"],
                ["gpu/tensornetwork:tensornetwork_permutation_optimized_f32:jax-cuda-eager"],
            )

    def test_effort_does_not_change_case_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            standard=self.plan(Path(tmp)/"standard")
            scan=self.plan(Path(tmp)/"scan",effort="scan")
            self.assertEqual(standard["selected"],scan["selected"])
            for path in scan["suite_files"]:
                for problem in yaml.safe_load(Path(path).read_text())["problems"]:
                    self.assertEqual((problem["run"]["warmups"],problem["run"]["runs"]),(1,3))

    def test_explicit_full_case_and_backend_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=self.plan(Path(tmp),"full",backend_override="pytorch-cuda",problem="dense_solve_f64_4096_rhs16")
            self.assertEqual(plan["selected"],["gpu/dense:dense_solve_f64_4096_rhs16:pytorch-cuda"])
            with self.assertRaisesRegex(ValueError,"no GPU cases"):
                self.plan(Path(tmp)/"bad",problem="dense_solve_f64_4096_rhs16")

    def test_unsupported_missing_and_unrecorded_samples_never_count_as_covered(self):
        plan=dict(manifest_version=1,coverage="quick",effort="scan",expected=["s:a:b","s:c:b","s:d:b"],
                  selected=["s:a:b","s:c:b","s:d:b"],unselected=[])
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

    def test_restricted_accounting_ignores_full_only_records(self):
        plan = dict(manifest_version=1, coverage="quick", effort="standard",
                    expected=["s:a:b"], selected=["s:a:b"], unselected=["s:c:b"])
        def row(problem):
            return json.dumps(dict(suite_id="s", problem_id=problem, backend="b", status="ok",
                                   verification=dict(status="passed"),
                                   execution=dict(notes=json.dumps({"sampling": {"batch_ms": [1.0]}}))))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text(row("a") + "\n" + row("c") + "\n")
            self.assertEqual(selection.account(plan, [path])["unexpected"], ["s:c:b"])
            restricted = selection.account(plan, [path], restrict=True)
            self.assertEqual(restricted["unexpected"], [])
            self.assertEqual(restricted["executed"], ["s:a:b"])

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
                plan = dict(manifest_version=1, coverage="quick", effort="scan",
                            expected=[key], selected=[key], unselected=[])
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

    def test_restrict_keeps_selected_duplicates_and_unknown_ids(self):
        key = "gpu/tensornetwork:a:b"
        plan = dict(manifest_version=1, coverage="quick", effort="scan",
                    expected=[key], selected=[key], unselected=["gpu/tensornetwork:full:b"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"records.jsonl"
            rows = [self.record(), self.record(), dict(self.record(), problem_id="full"),
                    dict(self.record(), problem_id="full"), dict(self.record(), problem_id="unknown")]
            path.write_text("".join(json.dumps(row)+"\n" for row in rows))
            status = selection.account(plan, [path], restrict=True)
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
                (root/"benchmarks/gpu/manifests").mkdir(parents=True)
                for name in ("dense", "einsum", "sparse", "tensornetwork"):
                    shutil.copy(ROOT/f"benchmarks/gpu/{name}.yaml", root/f"benchmarks/gpu/{name}.yaml")
                shutil.copy(ROOT/"benchmarks/gpu/manifests/routine.yaml", root/"benchmarks/gpu/manifests/routine.yaml")
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
                           GPU_BENCH_TIMESTAMP="20000101_000000", BENCH_COVERAGE="full",
                           GPU_BENCH_SUITE="benchmarks/gpu/dense.yaml",
                           GPU_BENCH_BACKENDS="pytorch-cuda", GPU_BENCH_PROBLEM="dense_matmul_f64_3072",
                           GPU_BENCH_BACKEND_SLEEP="0", GPU_BENCH_SKIP_LATEST="1",
                           GPU_BENCH_ROUTINE_CONTROLS="0")
                process = subprocess.run(["bash", str(root/"scripts/run_gpu_suite.sh")],
                                         env=env, capture_output=True, text=True, timeout=20)
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
