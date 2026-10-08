#!/usr/bin/env python3
"""Regression detector on synthetic report fixtures (tenferro-rs #1946 B5).

Covers missing cases, changed baselines, numerical/liveness/unsupported failures,
noisy data (INCONCLUSIVE), unbalanced order, A/A noise, and a deliberate timing
regression that only confirmation mode may call a regression.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
DETECTOR = ROOT / "scripts/detect_regressions.py"
BASE, CAND = "a" * 40, "b" * 40
HARNESS = "c" * 40
KEY_A, KEY_B = "bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t4", "chain3_f64_n4_einsum_alloc[faer]@t4"


def row(key, per_op, *, status="passed", workers=4, noise=0.0):
    case_id, provider = key.split("[")[0], key.split("[")[1].split("]")[0]
    samples = [int(per_op * (1 + noise * ((i % 3) - 1))) for i in range(9)]
    return {"case_key": key, "case_id": case_id, "provider": provider, "threads": 4,
            "workload": "batched_dot", "dtype": "f64", "worker_count": workers,
            "execution_mode": "Managed", "route": {"output": "allocating"},
            "status": status, "correctness": status, "operations_per_sample": 1,
            "samples_ns": samples if status == "passed" else [],
            "error": "numerical mismatch" if status == "failed" else None}


def write_run(path, rows, *, tenferro=BASE, manifest=3, expected=None, harness=HARNESS, dirty=False):
    path.mkdir(parents=True)
    keys = [r["case_key"] for r in rows]
    expected = expected or keys
    (path / "samples_t4.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    executed = [r["case_key"] for r in rows if r["status"] == "passed"]
    failed = [r["case_key"] for r in rows if r["status"] in ("failed", "timeout")]
    unsupported = [r["case_key"] for r in rows if r["status"] == "unsupported"]
    status = {"case_status_version": 1, "suite_id": "cpu/session_matrix", "manifest_version": manifest,
              "coverage": "quick", "effort": "scan", "selection_filter": None,
              "complete": not failed and set(executed + unsupported) == set(expected),
              "expected": expected, "selected": expected, "not_selected": [],
              "executed": executed, "unsupported": unsupported, "failed": failed,
              "missing": [k for k in expected if k not in keys], "noisy": [],
              "unexpected_rows": [], "duplicate_rows": []}
    (path / "case_status_t4.json").write_text(json.dumps(status))
    (path / "run_t4.yaml").write_text(yaml.safe_dump({
        "tenferro_rs": {"commit": tenferro, "dirty": False},
        "harness": {"commit": harness, "dirty": dirty},
        "collection": {"manifest_version": manifest}, "environment": {"hostname": "h"}}))
    return path


def declared_config(path, **overrides):
    config = yaml.safe_load((ROOT / "benchmarks/cpu/confirmation.yaml").read_text())
    config.update(status="declared", declared_before_candidate_results="2026-09-29T00:00:00Z",
                  library={"baseline_commit": BASE, "candidate_commit": CAND},
                  harness_commit=HARNESS, build={"profile": "release", "features": "native"},
                  host={"target_profile": "amd-cpu", "hostname": "h", "affinity": "none",
                        "provider": "faer"},
                  threads=[4], cases={"suite_id": "cpu/session_matrix", "manifest_version": 3,
                                      "coverage": "quick", "case_ids": None},
                  timing={"scope": "many_operations_single_interval", "cache_pool_state": "warm"},
                  repetitions={"warmups": 1, "runs": 9, "rounds": 2},
                  statistic="median_of_round_ratios",
                  thresholds={"relative": 0.05, "absolute_ns": 100},
                  noise={"max_cov": 0.05, "max_aa_relative_spread": 0.02,
                         "host_idle_guard": "enabled"})
    config.update(overrides)
    path.write_text(yaml.safe_dump(config))
    return path


def paired(root, arms, rows_for, *, order=None, tenferro=None):
    root.mkdir(parents=True)
    order = order or [(1, arms[0]), (1, arms[1]), (2, arms[1]), (2, arms[0])]
    entries = []
    for rnd, arm in order:
        name = f"r{rnd:02d}_{arm}"
        write_run(root / name, rows_for(arm, rnd), tenferro=(tenferro or {}).get(arm, BASE if arm != "candidate" else CAND))
        entries.append({"round": rnd, "arm": arm, "dir": name})
    (root / "plan.json").write_text(json.dumps({"order": entries, "threads": [4]}))
    return root


def run(*args):
    result = subprocess.run([sys.executable, str(DETECTOR), *map(str, args)],
                            capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


class ScanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_identical_runs_are_ok(self):
        b = write_run(self.tmp / "b", [row(KEY_A, 1000), row(KEY_B, 500)])
        c = write_run(self.tmp / "c", [row(KEY_A, 1000), row(KEY_B, 500)])
        code, out = run("scan", "--baseline", b, "--candidate", c)
        self.assertEqual(code, 0, out)
        self.assertIn("NO_SIGNAL", out)

    def test_deliberate_regression_is_only_a_suspicion_in_scan(self):
        b = write_run(self.tmp / "b", [row(KEY_A, 1000, noise=0.01)])
        c = write_run(self.tmp / "c", [row(KEY_A, 2000, noise=0.01)], tenferro=CAND)
        code, out = run("scan", "--baseline", b, "--candidate", c)
        self.assertEqual(code, 0, out)
        self.assertIn("SUSPECT_SLOWER", out)
        self.assertNotIn("| REGRESSION", out)

    def test_missing_case_is_deterministic(self):
        b = write_run(self.tmp / "b", [row(KEY_A, 1000), row(KEY_B, 500)])
        c = write_run(self.tmp / "c", [row(KEY_A, 1000)], expected=[KEY_A, KEY_B])
        code, out = run("scan", "--baseline", b, "--candidate", c)
        self.assertEqual(code, 2, out)
        self.assertIn("MISSING", out)

    def test_changed_baseline_is_deterministic(self):
        b = write_run(self.tmp / "b", [row(KEY_A, 1000)])
        c = write_run(self.tmp / "c", [row(KEY_A, 1000)], manifest=4)
        code, out = run("scan", "--baseline", b, "--candidate", c)
        self.assertEqual(code, 2, out)
        self.assertIn("CHANGED_BASELINE", out)

    def test_worker_or_provider_mismatch_is_deterministic(self):
        b = write_run(self.tmp / "b", [row(KEY_A, 1000), row(KEY_B, 500)])
        c = write_run(self.tmp / "c", [row(KEY_A, 1000, workers=1), row(KEY_B, 500)])
        code, out = run("scan", "--baseline", b, "--candidate", c)
        self.assertEqual(code, 2, out)
        self.assertIn("CHANGED_BASELINE", out)
        self.assertIn("worker_count", out)
        other = dict(row(KEY_B, 500), provider="blas")
        c2 = write_run(self.tmp / "c2", [row(KEY_A, 1000), other])
        code, out = run("scan", "--baseline", b, "--candidate", c2)
        self.assertEqual(code, 2, out)
        self.assertIn("provider", out)

    def test_numerical_liveness_and_unsupported_failures(self):
        b = write_run(self.tmp / "b", [row(KEY_A, 1000), row(KEY_B, 500)])
        bad = row(KEY_A, 0, status="failed")
        hung = dict(row(KEY_B, 0, status="timeout"), error="no result within 900 s")
        c = write_run(self.tmp / "c", [bad, hung])
        code, out = run("scan", "--baseline", b, "--candidate", c)
        self.assertEqual(code, 2, out)
        self.assertIn("NUMERICAL_FAILURE", out)
        self.assertIn("LIVENESS_FAILURE", out)
        c2 = write_run(self.tmp / "c2", [row(KEY_A, 0, status="unsupported"), row(KEY_B, 500)])
        code, out = run("scan", "--baseline", b, "--candidate", c2)
        self.assertEqual(code, 2, out)
        self.assertIn("NEWLY_UNSUPPORTED", out)

class ConfirmTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.aa = paired(self.tmp / "aa", ("a", "b"), lambda arm, r: [row(KEY_A, 1000, noise=0.01)],
                         tenferro={"a": BASE, "b": BASE})

    def confirm(self, pdir, config, aa=True):
        args = ["confirm", "--config", config, "--paired-dir", pdir]
        if aa:
            args += ["--aa-dir", self.aa]
        return run(*args)

    def test_aa_characterization_needs_no_thresholds(self):
        code, out = run("aa", "--aa-dir", self.aa)
        self.assertEqual(code, 0, out)
        self.assertIn("Balanced order: yes", out)
        self.assertIn(KEY_A, out)

    def test_placeholder_config_is_refused(self):
        pdir = paired(self.tmp / "p", ("baseline", "candidate"), lambda arm, r: [row(KEY_A, 1000)])
        code, out = self.confirm(pdir, ROOT / "benchmarks/cpu/confirmation.yaml")
        self.assertEqual(code, 2, out)
        self.assertIn("not declared", out)

    def test_deliberate_regression_is_confirmed(self):
        pdir = paired(self.tmp / "p", ("baseline", "candidate"),
                      lambda arm, r: [row(KEY_A, 1000 if arm == "baseline" else 1500, noise=0.01)])
        code, out = self.confirm(pdir, declared_config(self.tmp / "c.yaml"))
        self.assertEqual(code, 1, out)
        self.assertIn("| REGRESSION", out)

    def test_small_difference_is_no_change(self):
        pdir = paired(self.tmp / "p", ("baseline", "candidate"),
                      lambda arm, r: [row(KEY_A, 1000 if arm == "baseline" else 1010, noise=0.01)])
        code, out = self.confirm(pdir, declared_config(self.tmp / "c.yaml"))
        self.assertEqual(code, 0, out)
        self.assertIn("NO_CHANGE", out)

    def test_noisy_unbalanced_or_uncharacterized_runs_are_inconclusive(self):
        noisy = paired(self.tmp / "noisy", ("baseline", "candidate"),
                       lambda arm, r: [row(KEY_A, 1000 if arm == "baseline" else 1500, noise=0.3)])
        code, out = self.confirm(noisy, declared_config(self.tmp / "c.yaml"))
        self.assertEqual(code, 0, out)
        self.assertIn("INCONCLUSIVE", out)
        self.assertIn("CoV", out)
        unbalanced = paired(self.tmp / "unbalanced", ("baseline", "candidate"),
                            lambda arm, r: [row(KEY_A, 1000 if arm == "baseline" else 1500, noise=0.01)],
                            order=[(1, "baseline"), (1, "candidate"), (2, "baseline"), (2, "candidate")])
        code, out = self.confirm(unbalanced, declared_config(self.tmp / "c.yaml"))
        self.assertIn("not balanced", out)
        self.assertEqual(code, 0, out)
        code, out = self.confirm(unbalanced, declared_config(self.tmp / "c.yaml"), aa=False)
        self.assertIn("INCONCLUSIVE", out)
        wide = paired(self.tmp / "wide", ("a", "b"),
                      lambda arm, r: [row(KEY_A, 1000 if arm == "a" else 1100, noise=0.01)],
                      tenferro={"a": BASE, "b": BASE})
        good = paired(self.tmp / "good", ("baseline", "candidate"),
                      lambda arm, r: [row(KEY_A, 1000 if arm == "baseline" else 1500, noise=0.01)])
        code, out = run("confirm", "--config", declared_config(self.tmp / "c.yaml"),
                        "--paired-dir", good, "--aa-dir", wide)
        self.assertEqual(code, 0, out)
        self.assertIn("A/A spread", out)

    def test_commit_that_differs_from_the_declaration_is_a_changed_baseline(self):
        pdir = paired(self.tmp / "p", ("baseline", "candidate"), lambda arm, r: [row(KEY_A, 1000)],
                      tenferro={"baseline": "d" * 40, "candidate": CAND})
        code, out = self.confirm(pdir, declared_config(self.tmp / "c.yaml"))
        self.assertEqual(code, 2, out)
        self.assertIn("CHANGED_BASELINE", out)


if __name__ == "__main__":
    unittest.main()
