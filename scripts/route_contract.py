#!/usr/bin/env python3
"""Deterministic route-contract verdicts from `cpu_route_diagnostic` rows.

Contract (tenferro-rs #1946 B2/B3): for every case pair that differs only in
output ownership (allocating ``dot_general_read`` / ``einsum`` vs the ``_into``
twin), both routes must honor the same applicable batch policy:

* both reject the policy with a typed ``unsupported`` error, or both run;
* when both run and the provider was observed, both take the same outer
  fan-out lane decision (``lane_used``), and each route's provider batch
  counts cover the batch exactly once;
* both pass the numerical check.

Identical provider-call counts are *not* required: the 1-vs-4 call counts at
5a4e7fd84 are the symptom, not a permanent implementation contract.

A violated contract is a deterministic FAIL, independent of timing noise. A
pair with a missing twin is INCOMPLETE and also fails the run: absence never
counts as a pass. Lowering divergence between a batched einsum route and the
dot_general route of the same shape is reported as information only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_layout import safe_target_profile  # noqa: E402

PASS, FAIL, INCOMPLETE, UNSUPPORTED, NOT_OBSERVABLE = (
    "PASS", "FAIL", "INCOMPLETE", "UNSUPPORTED", "NOT_OBSERVABLE")
FAILING = {FAIL, INCOMPLETE}


def _route_summary(row: dict) -> dict:
    return {
        "case_id": row.get("case_id"),
        "status": row.get("status"),
        "provider_calls": row.get("provider_call_count"),
        "uninitialized_calls": row.get("uninitialized_calls"),
        "initialized_calls": row.get("initialized_calls"),
        "lane_used": row.get("lane_used"),
        "lane_calls": row.get("lane_calls"),
        "batch_sizes": row.get("batch_sizes"),
        "batch_total": row.get("batch_total"),
        "effective_policy": row.get("effective_policy"),
        "worker_count": row.get("worker_count"),
        "reason": row.get("unsupported_reason") or row.get("error"),
    }


def _covers_batch(row: dict) -> bool:
    calls = row.get("provider_calls") or []
    if any(c.get("entry") == "grouped" for c in calls):
        return True  # grouped jobs carry no batch count
    return row.get("batch_total") == row.get("batch")


def judge_pair(alloc: dict | None, into: dict | None) -> tuple[str, str]:
    if alloc is None or into is None:
        missing = "allocating" if alloc is None else "_into"
        return INCOMPLETE, f"{missing} twin has no diagnostic row"
    statuses = (alloc.get("status"), into.get("status"))
    if "failed" in statuses:
        return FAIL, "a route failed its numerical/execution check"
    if statuses == ("unsupported", "unsupported"):
        reason = alloc.get("unsupported_reason") or "typed Unsupported error"
        return UNSUPPORTED, f"both routes unsupported: {reason}"
    if "unsupported" in statuses:
        which = "allocating" if statuses[0] == "unsupported" else "_into"
        return FAIL, f"only the {which} route rejects the policy"
    if statuses != ("observed", "observed"):
        return FAIL, f"unexpected statuses {statuses}"
    if not alloc.get("provider_call_count") and not into.get("provider_call_count"):
        return NOT_OBSERVABLE, "no GEMM provider call observed on either route"
    if bool(alloc.get("lane_used")) != bool(into.get("lane_used")):
        return FAIL, (f"lane decision differs: allocating lane_used={alloc.get('lane_used')} "
                      f"({alloc.get('provider_call_count')} calls, batches "
                      f"{alloc.get('batch_sizes')}), _into lane_used={into.get('lane_used')} "
                      f"({into.get('provider_call_count')} calls, batches "
                      f"{into.get('batch_sizes')})")
    for name, row in (("allocating", alloc), ("_into", into)):
        if not _covers_batch(row):
            return FAIL, (f"{name} provider batches {row.get('batch_sizes')} do not cover "
                          f"batch {row.get('batch')} exactly once")
    return PASS, "same lane decision; batches covered exactly once"


def evaluate(rows: list[dict]) -> dict:
    pairs: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row.get("pair"):
            side = "into" if row.get("output") == "into" else "alloc"
            pairs[(row["pair"], int(row["requested_threads"]))][side] = row
    verdicts = []
    for (pair, threads), sides in sorted(pairs.items()):
        verdict, reason = judge_pair(sides.get("alloc"), sides.get("into"))
        verdicts.append({
            "pair": pair, "threads": threads, "verdict": verdict, "reason": reason,
            "allocating": _route_summary(sides["alloc"]) if "alloc" in sides else None,
            "into": _route_summary(sides["into"]) if "into" in sides else None,
        })
    # Informational: einsum lowering vs the dot_general route of the same shape.
    divergences = []
    lane_by_pair = {(v["pair"], v["threads"]): v for v in verdicts}
    for (pair, threads), v in lane_by_pair.items():
        if not pair.startswith("beinsum_"):
            continue
        twin = lane_by_pair.get(("bdot_" + pair.removeprefix("beinsum_"), threads))
        if twin and v["allocating"] and twin["allocating"]:
            a, b = v["allocating"].get("lane_used"), twin["allocating"].get("lane_used")
            if a != b:
                divergences.append({"pair": pair, "threads": threads,
                                    "einsum_lane_used": a, "dot_general_lane_used": b})
    failing = [v for v in verdicts if v["verdict"] in FAILING]
    return {
        "contract": "allocating and _into twins honor the same batch policy",
        "verdicts": verdicts,
        "counts": {k: sum(v["verdict"] == k for v in verdicts)
                   for k in (PASS, FAIL, INCOMPLETE, UNSUPPORTED, NOT_OBSERVABLE)},
        "lowering_divergences": divergences,
        "status": "FAIL" if failing else "PASS",
    }


def load_rows(run_dir: Path) -> list[dict]:
    return [json.loads(line) for path in sorted(run_dir.glob("diagnostics_t*.jsonl"))
            for line in path.read_text().splitlines() if line.strip()]


def render(result: dict, run_dir: Path, metadata: str = "") -> str:
    lines = ["# CPU route-contract diagnostics", "",
             "Mechanism diagnostics (provider spy + counting allocator), **not timing**. "
             "Verdicts are deterministic; see `scripts/route_contract.py` for the contract.", "",
             f"- Raw run: `{run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir}`",
             f"- Overall: **{result['status']}** — " + ", ".join(
                 f"{k} {v}" for k, v in result["counts"].items()), ""]
    if metadata:
        lines += [metadata.rstrip(), ""]
    lines += ["| Pair | Threads | Verdict | Allocating calls (uninit/init, lanes, batches) | "
              "_into calls (uninit/init, lanes, batches) | Reason |",
              "|---|---:|---|---|---|---|"]

    def side(s):
        if s is None:
            return "missing"
        if s["status"] != "observed":
            return f"{s['status']}"
        return (f"{s['uninitialized_calls']}/{s['initialized_calls']}, lanes "
                f"{s['lane_calls']}, {s['batch_sizes']}")

    for v in result["verdicts"]:
        lines.append(f"| `{v['pair']}` | {v['threads']} | {v['verdict']} | "
                     f"{side(v['allocating'])} | {side(v['into'])} | {v['reason']} |")
    if result["lowering_divergences"]:
        lines += ["", "## Lowering divergences (information, not a contract)", ""]
        for d in result["lowering_divergences"]:
            lines.append(f"- `{d['pair']}` t{d['threads']}: einsum lane_used="
                         f"{d['einsum_lane_used']}, dot_general lane_used="
                         f"{d['dot_general_lane_used']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-profile")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    result = evaluate(load_rows(run_dir))
    (run_dir / "route_contract.json").write_text(json.dumps(result, indent=2) + "\n")
    metadata = ""
    for path in sorted(run_dir.glob("run_t*.yaml")):
        metadata += f"- Metadata: `{path.name}`\n"
    text = render(result, run_dir, metadata)
    (run_dir / "report.md").write_text(text)
    if args.target_profile:
        latest = ROOT / "result" / safe_target_profile(args.target_profile) / "cpu/route_contract.md"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(text)
    print(f"route contract: {result['status']} {result['counts']}")
    return 3 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
