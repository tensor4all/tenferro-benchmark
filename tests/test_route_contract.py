#!/usr/bin/env python3
"""Route-contract verdicts: real baseline/repair counter rows and synthetic edge cases."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import route_contract as rc  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures/route_contract"
F2 = "bdot_f64_b1024_m4n4k4_direct_auto"


def load(name):
    return [json.loads(line) for line in (FIXTURES / name).read_text().splitlines()]


def verdict(result, pair, threads=4):
    return next(v for v in result["verdicts"] if v["pair"] == pair and v["threads"] == threads)


class RouteContractTest(unittest.TestCase):
    def test_baseline_5a4e7fd84_fails_the_f2_pair(self):
        # Counter rows recorded from cpu_route_diagnostic built against 5a4e7fd84.
        result = rc.evaluate(load("baseline_t4.jsonl"))
        self.assertEqual(result["status"], "FAIL")
        f2 = verdict(result, F2)
        self.assertEqual(f2["verdict"], rc.FAIL)
        self.assertIn("lane decision differs", f2["reason"])
        self.assertEqual(f2["allocating"]["uninitialized_calls"], 1)
        self.assertEqual(f2["into"]["lane_calls"], 4)
        # Below the cost cutoff both routes stay one call: the contract holds.
        self.assertEqual(verdict(result, "bdot_f64_b64_m4n4k4_direct_auto")["verdict"], rc.PASS)
        self.assertEqual(verdict(result, "bdot_f64_b1024_m4n4k4_direct_forced-whole-batch-vendor")
                         ["verdict"], rc.UNSUPPORTED)

    def test_repair_passes_the_same_pair(self):
        result = rc.evaluate(load("repair_t4.jsonl"))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(verdict(result, F2)["verdict"], rc.PASS)

    def test_missing_twin_is_incomplete_and_fails_the_run(self):
        rows = [r for r in load("repair_t4.jsonl") if not r["case_id"].endswith("_into_auto")
                or "b1024" not in r["case_id"]]
        result = rc.evaluate(rows)
        self.assertEqual(verdict(result, F2)["verdict"], rc.INCOMPLETE)
        self.assertEqual(result["status"], "FAIL")

    def test_one_sided_unsupported_and_uncovered_batch_fail(self):
        rows = load("repair_t4.jsonl")
        alloc = next(r for r in rows if r["case_id"] == "bdot_f64_b1024_m4n4k4_direct_alloc_auto")
        into = next(r for r in rows if r["case_id"] == "bdot_f64_b1024_m4n4k4_direct_into_auto")
        unsupported = dict(copy.deepcopy(into), status="unsupported", unsupported_reason="x")
        self.assertEqual(rc.judge_pair(alloc, unsupported)[0], rc.FAIL)
        short = copy.deepcopy(into)
        short["batch_total"] = 768
        short["batch_sizes"] = [256, 256, 256]
        self.assertEqual(rc.judge_pair(alloc, short)[0], rc.FAIL)
        failed = dict(copy.deepcopy(alloc), status="failed", error="numerical mismatch")
        self.assertEqual(rc.judge_pair(failed, into)[0], rc.FAIL)
        # Different call counts alone are allowed when the lane decision agrees.
        fewer = copy.deepcopy(alloc)
        fewer.update(provider_call_count=2, lane_calls=2, batch_sizes=[512, 512])
        fewer["provider_calls"] = fewer["provider_calls"][:2]
        self.assertEqual(rc.judge_pair(fewer, into)[0], rc.PASS)


if __name__ == "__main__":
    unittest.main()
