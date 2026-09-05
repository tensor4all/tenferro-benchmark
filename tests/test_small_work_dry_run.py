import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_small_work as runner


class DryRunTests(unittest.TestCase):
    def test_preview_filters_cases_observes_resources_and_never_executes(self):
        for variant in ("valid", "busy", "unsupported"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = SimpleNamespace(suite=ROOT / "benchmarks/cpu/small_work.yaml",
                    results_root=root / "data", target_profile="amd-cpu", output=root / "preview.json",
                    dry_run=True, observation_window=2.0, cpuset="0")
                selection = {"status": "valid" if variant == "valid" else "inconclusive",
                             "cpus": [0] if variant == "valid" else [],
                             "effective_threads": 1 if variant == "valid" else 0,
                             "reasons": [] if variant == "valid" else ["SMT sibling busy"]}
                stdout = io.StringIO()
                with mock.patch.dict(os.environ, {"SMALL_WORK_CASE_FILTER": "add_f64_concrete_fresh"}), \
                     mock.patch.object(runner, "_select_live_resources", return_value=selection,
                                       side_effect=AttributeError("no affinity API") if variant == "unsupported" else None) as observe, \
                     mock.patch.object(runner, "verify_canonical_binding", side_effect=AssertionError("binding must not run")), \
                     mock.patch.object(runner, "prepare_cargo", side_effect=AssertionError("build must not run")), \
                     mock.patch.object(runner, "run_sequential", side_effect=AssertionError("benchmark must not run")), \
                     mock.patch.object(runner.os, "sched_setaffinity", create=True, side_effect=AssertionError("must not pin")), \
                     contextlib.redirect_stdout(stdout):
                    self.assertEqual(runner._suite_run(args), 0)
                result = json.loads(stdout.getvalue())
                self.assertEqual(result, json.loads(args.output.read_text()))
                self.assertEqual(result["status"], "DRY_RUN" if variant == "valid" else "INCONCLUSIVE")
                self.assertEqual([case["id"] for case in result["cases"]], ["add_f64_concrete_fresh"])
                self.assertEqual(result["commands"], [])
                self.assertIn("correctness", result["not_checked"])
                self.assertIn("build_provenance", result["not_checked"])
                observe.assert_called_once_with(1, .2, 2.0, "0")
                if variant == "unsupported":
                    self.assertIn("AttributeError", result["resource"]["reasons"][0])

    def test_invalid_filter_fails_before_resource_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(suite=ROOT / "benchmarks/cpu/small_work.yaml",
                results_root=Path(directory), target_profile="amd-cpu", output=None, dry_run=True)
            stdout = io.StringIO()
            with mock.patch.dict(os.environ, {"SMALL_WORK_CASE_FILTER": "no_such_case"}), \
                 mock.patch.object(runner, "_select_live_resources") as observe, \
                 contextlib.redirect_stdout(stdout):
                self.assertEqual(runner._suite_run(args), 1)
            observe.assert_not_called()
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["commands"], [])
            self.assertTrue(result["errors"])

    def test_cli_needs_no_binary_and_does_not_publish_latest_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [sys.executable, str(ROOT / "scripts/run_small_work.py"), "--dry-run",
                       "--suite", str(ROOT / "benchmarks/cpu/small_work.yaml"),
                       "--results-root", str(root / "data"), "--result-root", str(root / "latest"),
                       "--observation-window", "0.01"]
            env = dict(os.environ)
            env.pop("SMALL_WORK_CASE_FILTER", None)
            child = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(child.returncode, 0, child.stderr)
            result = json.loads(child.stdout)
            suite = runner.yaml.safe_load((ROOT / "benchmarks/cpu/small_work.yaml").read_text())
            self.assertEqual([case["id"] for case in result["cases"]], [case["id"] for case in suite["cases"]])
            self.assertEqual(result["commands"], [])
            self.assertFalse((root / "latest").exists())
            self.assertFalse((root / "small_work_run.json").exists())
            for flag in ("--prepare", "--component-probe", "--correctness-only", "--synthetic-fixture"):
                rejected = subprocess.run(command + [flag], cwd=root, env=env,
                                          capture_output=True, text=True, timeout=20)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("cannot be combined", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
