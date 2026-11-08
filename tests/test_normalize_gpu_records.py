"""Folding a family's raw rows into the routine record contract."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import yaml

import normalize_gpu_records as normalize
from normalize_gpu_records import compile_accounting, fold
from scripts.ci.check_gpu_run import validate_records
import gpu_routine_selection as selection


def family_row(**overrides):
    row = {"schema_version": 1, "suite_id": "gpu/lstsq", "case_id": "lstsq_tall_f64_s256_rhs8",
           "op": "lstsq", "backend": "pytorch-cuda", "status": "ok",
           "timing": {"operations_per_sample": 4, "batch_ms": [0.4, 0.5, 0.45], "warmups": 1},
           "verification": {"max_error_over_tolerance": 1e-3, "reference": "numpy/scipy"}}
    row.update(overrides)
    return row


def permutation_row(**overrides):
    row = {"schema_version": 1, "suite_id": "gpu/permutation", "runner": "python",
           "pattern_id": "reverse_18d_3", "label": "reverse 2d", "backend": "pytorch-cuda",
           "dtype": "f64", "status": "ok", "correctness": "passed", "warmup": 3, "iters": 7,
           "median_ms": 0.0123, "p25_ms": 0.011, "p75_ms": 0.013}
    row.update(overrides)
    return row


class NormalizeTest(unittest.TestCase):
    def fold_and_validate(self, row, suite_id=None):
        folded = fold(row, suite_id)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text(json.dumps(folded) + "\n")
            validate_records(path, folded["suite_id"],
                             {(folded["problem_id"], folded["backend"])})
        return folded

    def test_batched_samples_become_a_per_operation_median(self):
        folded = self.fold_and_validate(family_row())
        # 0.4/4, 0.5/4, 0.45/4 ms per operation.
        self.assertAlmostEqual(folded["timing"]["median_ms"], 0.1125)
        self.assertEqual(folded["timing"]["timed_runs"], 3)
        self.assertEqual(folded["verification"]["status"], "passed")

    def test_single_sample_rows_keep_their_median_and_recorded_iqr(self):
        folded = self.fold_and_validate(permutation_row())
        self.assertAlmostEqual(folded["timing"]["median_ms"], 0.0123)
        self.assertAlmostEqual(folded["timing"]["iqr_ms"], 0.002)
        self.assertEqual(folded["verification"]["status"], "passed")

    def test_unverified_and_failed_rows_never_look_measured(self):
        failed = fold(permutation_row(correctness="mismatch"))
        self.assertEqual(failed["verification"]["status"], "failed")
        # The publication gate refuses such a row rather than publishing it.
        with self.assertRaisesRegex(ValueError, "verification did not pass"):
            self.fold_and_validate(permutation_row(correctness="mismatch"))
        unsupported = fold(family_row(status="unsupported", reason="output reuse unsupported",
                                      verification={"max_error_over_tolerance": 0.5}))
        self.assertEqual(unsupported["status"], "unsupported")
        self.assertEqual(unsupported["execution"]["unsupported_reason"], "output reuse unsupported")

    def test_a_skipped_correctness_is_never_reported_as_a_numerical_failure(self):
        # The permutation runners put the explanation in `notes`, not `reason`,
        # and a cell whose arm never ran carries correctness=skipped. Folding
        # that to `failed` is how a "not measured" cell came to be printed as
        # `numerical verification failed 9/9` on a host without cuTENSOR.
        notes = "failed to load cuTENSOR library (tried libcutensor.so.2)"
        row = permutation_row(status="skipped", correctness="skipped", median_ms=None,
                              p25_ms=None, p75_ms=None, notes=notes)
        folded = fold(row)
        self.assertEqual(folded["verification"]["status"], "skipped")
        self.assertEqual(folded["verification"]["reason"], notes)
        self.assertEqual(folded["execution"]["unsupported_reason"], notes)
        self.assertIsNone(folded["timing"]["median_ms"])
        key = "gpu/permutation:reverse_18d_3:pytorch-cuda"
        plan = dict(manifest_version=2, effort="scan", expected=[key], selected=[key])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text(json.dumps(folded) + "\n")
            status = selection.account(plan, [path])
        self.assertEqual(status["skipped"], [key])
        self.assertEqual(status["failed"], [])
        self.assertEqual(status["reasons"][key], notes)

    def test_a_row_without_a_case_identity_is_rejected(self):
        row = family_row()
        del row["case_id"]
        with self.assertRaisesRegex(ValueError, "without a case identity"):
            fold(row)

    def test_a_row_without_a_suite_identity_takes_the_run_suite(self):
        # The public API families write case_id rows without suite_id; run.yaml
        # names the suite the run belongs to.
        row = family_row()
        del row["suite_id"]
        folded = self.fold_and_validate(row, "gpu/lstsq")
        self.assertEqual(folded["suite_id"], "gpu/lstsq")
        # A row that does carry one keeps it, so an emitter-side suite_id wins.
        self.assertEqual(fold(family_row(), "gpu/isolated_ops")["suite_id"], "gpu/lstsq")
        with self.assertRaisesRegex(ValueError, "without a suite identity"):
            fold(row)

    def test_a_python_arms_compile_is_not_summed_into_the_nvrtc_total(self):
        # jax-cuda-jit records a real compile interval but no NVRTC module, so a warm
        # CUDA cache must not be reported as having paid NVRTC time.
        raw = [
            {"backend": "direct", "compile_modules": 1, "overhead": {"case_wall_ms": 1000.0},
             "timing": {"compile_time_ms": 700.0}},
            {"backend": "jax-cuda-jit", "overhead": {"case_wall_ms": 1000.0},
             "timing": {"compile_time_ms": 2000.0}},
        ]
        self.assertEqual(compile_accounting(raw)["compile_time_ms"], 700.0)
        self.assertEqual(compile_accounting(raw)["modules_compiled"], 1)

    def test_a_compiling_case_without_sampling_still_contributes(self):
        # An excluded or failed case that compiled keeps its time at the top level because
        # a null timing block has nowhere else to put it.
        raw = [{"backend": "direct", "compile_modules": 3, "compile_time_ms": 5000.0,
                "timing": None, "overhead": {"case_wall_ms": 5000.0}}]
        accounting = compile_accounting(raw)
        self.assertEqual(accounting["modules_compiled"], 3)
        self.assertEqual(accounting["compile_time_ms"], 5000.0)

    def test_normalizing_records_the_totals_and_keeps_the_rest_of_the_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run = tmp / "run.yaml"
            run.write_text("suite_id: gpu/linalg_public_api_throughput\n"
                           "cuda:\n  artifact_cache:\n    regime: cold\n")
            arm = tmp / "direct.jsonl"
            arm.write_text(json.dumps(family_row(
                backend="direct", compile_cache="miss", compile_modules=2,
                overhead={"case_wall_ms": 4000.0},
                timing={"operations_per_sample": 1, "batch_ms": [2.0],
                        "compile_time_ms": 3000.0})) + "\n")
            out = tmp / "records.jsonl"
            argv = ["normalize_gpu_records.py", "--output", str(out),
                    "--run-metadata", str(run), str(arm)]
            with patch.object(sys, "argv", argv):
                normalize.main()
            metadata = yaml.safe_load(run.read_text())
            self.assertEqual(metadata["collection"]["compile"],
                             {"case_wall_seconds": 4.0, "compiling_processes": 1,
                              "modules_compiled": 2, "compile_time_ms": 3000.0,
                              "compile_time_scope": normalize.COMPILE_TIME_SCOPE,
                              "untimed_wall_seconds": 3.0})
            # Everything the collector wrote is still there.
            self.assertEqual(metadata["cuda"]["artifact_cache"]["regime"], "cold")
            once = run.read_text()
            with patch.object(sys, "argv", argv):
                normalize.main()
            self.assertEqual(run.read_text(), once)
            record = json.loads(out.read_text().splitlines()[0])
            self.assertEqual(record["timing"]["compile_cache"], "miss")
            self.assertEqual(record["timing"]["compile_modules"], 2)

    def test_compile_accounting_reaches_the_record(self):
        folded = fold(family_row(compile_cache="miss", compile_modules=2,
                                 timing={"operations_per_sample": 4, "batch_ms": [0.4],
                                         "compile_time_ms": 69014.2}), "gpu/linalg_public_api_throughput")
        self.assertEqual(folded["timing"]["compile_cache"], "miss")
        self.assertEqual(folded["timing"]["compile_modules"], 2)
        self.assertEqual(folded["timing"]["compile_time_ms"], 69014.2)
        # A family that does not compile keeps records without the fields.
        self.assertNotIn("compile_cache", fold(family_row())["timing"])

    def test_compile_accounting_sums_what_the_arm_files_recorded(self):
        raw = [
            {"backend": "direct", "compile_modules": 3, "overhead": {"case_wall_ms": 4000.0},
             "timing": {"compile_time_ms": 2000.0}},
            {"backend": "direct", "compile_modules": 0, "overhead": {"case_wall_ms": 1000.0},
             "timing": {"compile_time_ms": 0.0}},
            {"backend": "eager", "compile_modules": 1, "overhead": {"case_wall_ms": 2200.4},
             "timing": {"compile_time_ms": 2000.7}},
        ]
        self.assertEqual(compile_accounting(raw), {
            "case_wall_seconds": 7.2, "compiling_processes": 2,
            "modules_compiled": 4, "compile_time_ms": 4000.7,
            "compile_time_scope": normalize.COMPILE_TIME_SCOPE,
            "untimed_wall_seconds": 4.001})

    def test_a_warm_run_records_zeros_rather_than_no_block(self):
        raw = [{"backend": "direct", "compile_modules": 0,
                "overhead": {"case_wall_ms": 500.0}, "timing": {"compile_time_ms": 0.0}}]
        self.assertEqual(compile_accounting(raw), {
            "case_wall_seconds": 0.5, "compiling_processes": 0,
            "modules_compiled": 0, "compile_time_ms": 0.0,
            "compile_time_scope": normalize.COMPILE_TIME_SCOPE,
            "untimed_wall_seconds": 0.0})

    def test_an_arm_that_measures_no_compile_interval_keeps_the_wall_as_a_wall(self):
        # The public-API Rust arms publish compile_time_ms null and the case's untimed wall
        # under overhead, so the accounting must not sum a missing interval into a total and
        # must say what the field it does publish is.
        raw = [{"backend": "direct", "compile_modules": 2, "compile_time_ms": None,
                "overhead": {"case_wall_ms": 4000.0, "untimed_ms": 3000.0},
                "timing": {"compile_time_ms": None}},
               {"backend": "direct", "compile_modules": 0, "compile_time_ms": None,
                "overhead": {"case_wall_ms": 1000.0, "untimed_ms": 990.0},
                "timing": {"compile_time_ms": None}}]
        self.assertEqual(compile_accounting(raw), {
            "case_wall_seconds": 5.0, "compiling_processes": 1,
            "modules_compiled": 2, "compile_time_ms": None,
            "compile_time_scope": normalize.COMPILE_TIME_ABSENT,
            "untimed_wall_seconds": 3.0})

    def test_every_folded_record_says_what_its_compile_field_is(self):
        folded = fold(family_row(compile_cache="miss", compile_modules=2,
                                 timing={"operations_per_sample": 1, "batch_ms": [0.4],
                                         "compile_time_ms": None}), "gpu/linalg_public_api_throughput")
        self.assertIsNone(folded["timing"]["compile_time_ms"])
        self.assertEqual(folded["timing"]["compile_time_scope"], normalize.COMPILE_TIME_ABSENT)
        kept = fold(family_row(compile_cache="miss", compile_modules=2,
                               timing={"operations_per_sample": 1, "batch_ms": [0.4],
                                       "compile_time_ms": 3000.0}), "gpu/linalg_public_api_throughput")
        self.assertEqual(kept["timing"]["compile_time_ms"], 3000.0)
        self.assertEqual(kept["timing"]["compile_time_scope"], normalize.COMPILE_TIME_SCOPE)

    def test_a_jax_compile_and_an_nvrtc_wall_get_different_labels(self):
        # #363: the legacy-wall label belongs to NVRTC records only. A jax-cuda-jit record
        # carries a real lowering/compile interval and no compile_modules.
        nvrtc = fold(family_row(backend="direct", compile_cache="miss", compile_modules=2,
                                timing={"operations_per_sample": 1, "batch_ms": [0.4],
                                        "compile_time_ms": 3000.0}), "gpu/isolated_ops")
        jax = fold(family_row(backend="jax-cuda-jit",
                              timing={"operations_per_sample": 1, "batch_ms": [0.4],
                                      "compile_time_ms": 250.0}), "gpu/isolated_ops")
        self.assertEqual(nvrtc["timing"]["compile_time_scope"], normalize.COMPILE_TIME_SCOPE)
        self.assertEqual(jax["timing"]["compile_time_scope"], normalize.FRAMEWORK_COMPILE_SCOPE)
        self.assertNotEqual(nvrtc["timing"]["compile_time_scope"],
                            jax["timing"]["compile_time_scope"])
        self.assertEqual(jax["timing"]["compile_time_ms"], 250.0)
        # The framework record still has no NVRTC fields and stays out of the NVRTC totals.
        self.assertNotIn("compile_modules", jax["timing"])

    def test_a_framework_arm_without_a_compile_interval_is_absent(self):
        folded = fold(family_row(backend="jax-cuda",
                                 timing={"operations_per_sample": 1, "batch_ms": [0.4],
                                         "compile_time_ms": None}), "gpu/isolated_ops")
        self.assertEqual(folded["timing"]["compile_time_scope"], normalize.COMPILE_TIME_ABSENT)

    def test_a_family_that_does_not_compile_records_nothing(self):
        self.assertIsNone(compile_accounting([family_row()]))

    def test_verification_time_comes_from_the_phase_block_when_timing_omits_it(self):
        # The public API families measure it and publish it as `overhead.verify_ms`;
        # the contract field must carry it so a reader does not need to know that.
        folded = self.fold_and_validate(family_row(
            overhead={"timed_ms": 12.0, "case_wall_ms": 1012.0, "untimed_ms": 1000.0,
                      "fixture_ms": 15.0, "verify_ms": 940.0}))
        self.assertEqual(folded["timing"]["verification_time_ms"], 940.0)
        self.assertEqual(json.loads(folded["execution"]["notes"])["overhead"]["verify_ms"], 940.0)

    def test_a_recorded_contract_value_wins_over_the_phase_block(self):
        folded = self.fold_and_validate(family_row(
            timing={"operations_per_sample": 1, "batch_ms": [0.4], "verification_time_ms": 3.5},
            overhead={"case_wall_ms": 10.0, "verify_ms": 9.0}))
        self.assertEqual(folded["timing"]["verification_time_ms"], 3.5)

    def test_a_row_without_a_phase_block_records_no_phases(self):
        folded = self.fold_and_validate(family_row())
        self.assertIsNone(folded["timing"]["verification_time_ms"])
        self.assertNotIn("overhead", json.loads(folded["execution"]["notes"]))


if __name__ == "__main__":
    unittest.main()
