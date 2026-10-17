import json
from pathlib import Path
import sys
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
            path.write_text(json.dumps(dict(suite_id="s",problem_id="a",backend="b",status="unsupported"))+"\n"+
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


if __name__ == "__main__": unittest.main()
