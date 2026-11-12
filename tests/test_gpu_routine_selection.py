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
            self.assertEqual(len(plan["selected"]), 708)
            self.assertEqual(plan["expected"], plan["selected"])
            for suite in ("dense", "einsum", "sparse", "tensornetwork", "elementwise",
                          "linalg_ad_latency", "linalg_jvp_vjp", "permutation", "lstsq",
                          "linalg_public_api_throughput", "fft", "nn", "structural", "isolated_ops",
                          "reuse_contraction"):
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
            self.assertFalse(any(key.endswith(":jax-cuda-eager") or key.endswith(":jax-cuda-jit")
                                 or "cublaslt" in key or "cutlass" in key
                                 for key in plan["selected"]))
            self.assertFalse(plan["vendor_arms"])
            # The permutation runners fix their own arms, cuTENSOR included.
            self.assertTrue(any(key.startswith("gpu/permutation:") and key.endswith(":cutensor")
                                for key in plan["selected"]))
            # A pattern runs only the arms it declares as participants, so the
            # suite is 44 rows, not the plain 10 x 6 product (#368).
            permutation = [key for key in plan["selected"] if key.startswith("gpu/permutation:")]
            self.assertEqual(len(permutation), 44)
            self.assertTrue(all(key.endswith(":memcpy-d2d") for key in permutation
                                if key.startswith("gpu/permutation:memcpy_24d_64x2:")))

    def test_vendor_arms_are_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(Path(tmp), vendor_arms=True)
            self.assertEqual(len(plan["selected"]), 872)
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
            self.assertEqual(plan["suite_drivers"][(Path(tmp)/"gpu_lstsq.yaml").as_posix()], "gpu_public_api")
            self.assertEqual(plan["suite_drivers"][(Path(tmp)/"gpu_permutation.yaml").as_posix()],
                             "gpu_permutation")

    def test_instance_driven_suites_hand_the_collector_a_curated_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(Path(tmp))
            instances = plan["suite_instances"]
            self.assertEqual(len(instances), 8)
            lstsq = yaml.safe_load(Path(instances[(Path(tmp)/"gpu_lstsq.yaml").as_posix()]).read_text())
            self.assertEqual([case["id"] for case in lstsq["cases"]],
                             ["lstsq_tall_f64_s256_rhs8"])
            self.assertEqual(lstsq["backends"],
                             ["tenferro-cuda-direct", "tenferro-cuda-eager", "pytorch-cuda", "jax-cuda"])
            patterns = yaml.safe_load(Path(instances[(Path(tmp)/"gpu_permutation.yaml").as_posix()]).read_text())
            self.assertEqual(len(patterns["patterns"]), 10)
            # The suite declaration names the materialized document, never the repo one.
            declaration = yaml.safe_load((Path(tmp)/"gpu_lstsq.yaml").read_text())
            self.assertEqual(declaration["problems"]["source"],
                             instances[(Path(tmp)/"gpu_lstsq.yaml").as_posix()])
            self.assertEqual(declaration["problems"]["include"], ["lstsq_tall_f64_s256_rhs8"])

    def test_excluded_ids_are_not_handed_to_the_collector(self):
        # The policy drops `exclude`d ids; the collector must hand over the same
        # set, or the report contains a case its own manifest does not declare.
        with tempfile.TemporaryDirectory() as tmp:
            instances = Path(tmp) / "instances.json"
            instances.write_text(json.dumps({"cases": [{"id": "keep", "n": 1048576},
                                                      {"id": "drop", "n": 1024}]}))
            declaration = Path(tmp) / "gpu_fft.yaml"
            declaration.write_text(yaml.safe_dump({
                "schema_version": 1,
                "suite_id": "gpu/fft",
                "title": "t",
                "description": "d",
                "backends": ["tenferro-cuda-trace"],
                "problems": {
                    "source": str(instances),
                    "include": ["keep", "drop"],
                    "exclude": ["drop"],
                    "regime": "throughput",
                },
            }))
            payload = selection.instances_for(declaration, Path(tmp) / "out.json")
        self.assertEqual([case["id"] for case in payload["cases"]], ["keep"])

    def test_effort_does_not_change_case_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            standard=self.plan(Path(tmp)/"standard")
            scan=self.plan(Path(tmp)/"scan",effort="scan")
            self.assertEqual(standard["selected"],scan["selected"])
            for path in scan["suite_files"]:
                problems = yaml.safe_load(Path(path).read_text())["problems"]
                if not isinstance(problems, list):
                    # Instance-driven suites take repetitions from the environment.
                    continue
                for problem in problems:
                    self.assertEqual((problem["run"]["warmups"],problem["run"]["runs"]),(1,3))

    def test_case_outside_the_routine_set_is_not_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=self.plan(Path(tmp),backend_override="pytorch-cuda",problem="dense_matmul_f64_3072")
            self.assertEqual(plan["selected"],["gpu/dense:dense_matmul_f64_3072:pytorch-cuda"])
            with self.assertRaisesRegex(ValueError,"no GPU cases"):
                self.plan(Path(tmp)/"bad",problem="dense_solve_f64_4096_rhs16")

    def test_a_named_probe_case_plans_its_own_rows(self):
        # n=4096 AD rows are declared beside the routine set, not in it: a named
        # probe must plan the rows it asked for (three arms, one case) while the
        # weekly set keeps planning only n=256.
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(Path(tmp), problem="linalg_ad_large_grad_sum_qr_jvp_f64_4096")
            expected = [
                f"gpu/linalg_jvp_vjp:linalg_ad_large_grad_sum_qr_jvp_f64_4096:{arm}"
                for arm in ("tenferro-cuda-trace", "tenferro-cuda-eager",
                            "pytorch-cuda", "jax-cuda")
            ]
            self.assertEqual(plan["selected"], expected)
            # The probe's rows join this plan's expected set, so packaging can
            # treat them as planned rather than `unexpected`.
            self.assertEqual([key for key in plan["expected"] if "4096" in key], expected)
            self.assertEqual(len(plan["expected"]), 708 + len(expected))
            self.assertEqual(plan["problem_filter"], "linalg_ad_large_grad_sum_qr_jvp_f64_4096")
            materialized = yaml.safe_load(Path(plan["suite_files"][0]).read_text())
            self.assertEqual([item["id"] for item in materialized["problems"]],
                             ["linalg_ad_large_grad_sum_qr_jvp_f64_4096"])

    def test_a_comma_separated_probe_selects_each_named_case(self):
        # Two dispatches run the 4096 matrix without ten separate pods: the
        # filter names every case, and each becomes part of this plan's rows.
        names = (
            "linalg_ad_large_grad_sum_qr_jvp_f64_4096,"
            "linalg_ad_large_grad_sum_lu_vjp_f64_4096"
        )
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(Path(tmp), problem=names)
            selected = [key for key in plan["selected"] if "4096" in key]
            self.assertEqual(len(selected), 8)  # two cases x four arms
            self.assertEqual({key.split(":")[1] for key in selected},
                             set(names.split(",")))
            materialized = yaml.safe_load(Path(plan["suite_files"][0]).read_text())
            self.assertEqual({item["id"] for item in materialized["problems"]},
                             set(names.split(",")))

    def test_probe_cases_stay_out_of_the_weekly_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(Path(tmp))
            self.assertFalse([key for key in plan["expected"] if key.endswith("_4096")])
            self.assertFalse([key for key in plan["selected"] if key.endswith("_4096")])
            self.assertEqual(len(plan["selected"]), 708)

    def test_a_stale_probe_case_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT/"benchmarks/gpu", root/"benchmarks/gpu")
            path = root/"benchmarks/gpu/manifests/routine.yaml"
            manifest = yaml.safe_load(path.read_text())
            manifest["suites"]["gpu/linalg_jvp_vjp"]["probe_problems"].append(
                "linalg_ad_large_grad_sum_absent_f64_4096")
            path.write_text(yaml.safe_dump(manifest, sort_keys=False))
            with self.assertRaisesRegex(ValueError, "stale routine problem manifest"):
                selection.materialize(path, ["benchmarks/gpu/linalg_jvp_vjp.yaml"], root/"out")

    def test_a_filtered_probe_leaves_out_the_suite_whose_runner_takes_no_filter(self):
        # gpu/permutation's runner emits its own arms over its own patterns and
        # reads neither GPU_BENCH_BACKENDS nor GPU_BENCH_PROBLEM. A plan that
        # narrowed it collected 35 rows it called `unexpected`, which refuses the
        # whole bundle, so a filtered selection must drop the suite rather than
        # narrow it (#385).
        with tempfile.TemporaryDirectory() as tmp:
            unfiltered = self.plan(Path(tmp)/"all")
            self.assertEqual(len([key for key in unfiltered["selected"]
                                  if key.startswith("gpu/permutation:")]), 44)
            self.assertEqual(unfiltered["skipped_unfiltered_suites"], [])
            for overrides in (dict(backend_override="tenferro-cuda-trace,tenferro-cuda-eager,pytorch-cuda"),
                              dict(backend_override="pytorch-cuda"),
                              dict(problem="dense_matmul_f64_3072")):
                with self.subTest(**overrides):
                    plan = self.plan(Path(tmp)/f"filtered-{len(str(overrides))}", **overrides)
                    self.assertFalse([key for key in plan["selected"]
                                      if key.startswith("gpu/permutation:")])
                    self.assertEqual(plan["skipped_unfiltered_suites"], ["gpu/permutation"])
                    # A filter never narrows a fixed-arm suite's own arms, so its
                    # 44 rows stay in `expected`; only the selection drops them.
                    self.assertEqual(len([key for key in plan["expected"]
                                          if key.startswith("gpu/permutation:")]), 44)
                    self.assertFalse([path for path in plan["suite_files"]
                                      if Path(path).name == "gpu_permutation.yaml"])
            # Asking only for the suite a filter cannot express says why, instead
            # of reporting an empty selection as if the manifest were broken.
            with self.assertRaisesRegex(ValueError, "gpu/permutation takes no arm or problem filter"):
                selection.materialize(MANIFEST, ["benchmarks/gpu/permutation.yaml"],
                                      Path(tmp)/"only-perm", backend_override="pytorch-cuda")

    def test_cli_announces_the_suite_a_filter_drops_on_stderr(self):
        # The notice must not reach stdout: run_gpu_suite.sh reads stdout with
        # `mapfile` as the `suite_path driver instances` row stream, so a stray
        # line there is taken as a suite path and aborts before measuring.
        with tempfile.TemporaryDirectory() as tmp:
            process = subprocess.run(
                [sys.executable, str(ROOT/"scripts/gpu_routine_selection.py"),
                 "--output", tmp, "--backends", "pytorch-cuda", *manifest_suites()],
                capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
            self.assertIn("skipped gpu/permutation", process.stderr)
            rows = [line for line in process.stdout.splitlines() if line.strip()]
            self.assertFalse([line for line in rows if line.startswith("skipped")], process.stdout)
            self.assertEqual(len(rows), len(manifest_suites())-1)
            for line in rows:
                self.assertEqual(len(line.split()), 3, line)

    def test_an_empty_selection_names_only_a_requested_suite(self):
        # Asking dense for a case it does not contain must not blame the fixed-arm
        # suite the probe never requested: that reports a cause the caller cannot
        # act on and hides the real one (no case matched the filter).
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "^no GPU cases selected$"):
                selection.materialize(MANIFEST, ["benchmarks/gpu/dense.yaml"], Path(tmp),
                                      problem="not-a-case")

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

    def test_a_nonpositive_sample_fails_the_case_and_says_why(self):
        plan=dict(manifest_version=2,effort="scan",expected=["s:a:b"],selected=["s:a:b"])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"records.jsonl"
            path.write_text(json.dumps(dict(suite_id="s",problem_id="a",backend="b",status="ok",
                                            verification=dict(status="passed"),
                                            execution=dict(notes=json.dumps({"sampling": {"batch_ms": [1.0, -1.0]}}))))+"\n")
            result=selection.account(plan,[path])
            self.assertEqual(result["failed"],["s:a:b"])
            # A failed case without a reason cannot be reported; the accounting
            # used to append it and move on.
            self.assertIn("s:a:b", result["reasons"])
            self.assertTrue(result["reasons"]["s:a:b"].strip())

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
                shutil.copytree(ROOT/"data/instances", root/"data/instances")
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
