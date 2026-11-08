#!/usr/bin/env python3
"""Compare the recorded output invariants of the arms that measured one case.

Every arm records `execution.notes.fingerprints`: sums, the L2 mass, the peak and
a seeded dot product of what it actually produced. Two arms computing the same
case must agree on all of them, and comparing those numbers costs a handful of
floats instead of a CPU reference plus a full-output comparison.

This is a diagnostic step for now: it reports disagreements and does not change a
row's verification status.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

#: The invariants, and the weight each one carries in a disagreement report.
FIELDS = (
    "sum_re", "sum_im", "sum_abs", "sum_squares",
    "max_abs", "dot_seeded_re", "dot_seeded_im",
)
#: A pair of arms agreeing to this relative difference is treated as agreeing.
#: Sums cancel, so their tolerance is relative to `sum_abs`, not to themselves.
DEFAULT_RTOL = 1.0e-9


def _relative(a: float, b: float, scale: float) -> float:
    denominator = max(abs(scale), 1e-300)
    return abs(a - b) / denominator


def compare_outputs(left: dict, right: dict, *, rtol=DEFAULT_RTOL) -> list[str]:
    """Fields on which two arms' invariants disagree beyond `rtol`."""
    if left.get("elements") != right.get("elements"):
        return [f"elements {left.get('elements')} != {right.get('elements')}"]
    problems = []
    scale = max(left.get("sum_abs") or 0.0, right.get("sum_abs") or 0.0)
    mass = max(left.get("sum_squares") or 0.0, right.get("sum_squares") or 0.0) ** 0.5
    for field in FIELDS:
        a, b = left.get(field), right.get(field)
        if a is None or b is None:
            problems.append(f"{field} missing")
            continue
        if field.startswith("sum_") and field not in ("sum_squares",):
            base = scale
        elif field == "sum_squares":
            base = mass
        elif field == "max_abs":
            base = max(abs(a), abs(b))
        else:  # seeded dot: bounded by the mass times the weight vector's norm
            base = mass * max(left.get("elements") or 1, 1) ** 0.5
        if _relative(a, b, base) > rtol:
            problems.append(f"{field} {a!r} != {b!r} (relative {_relative(a, b, base):.3e})")
    return problems


def fingerprints_of(record) -> list[dict] | None:
    notes = (record.get("execution") or {}).get("notes")
    if not isinstance(notes, str):
        return None
    try:
        document = json.loads(notes)
    except ValueError:
        return None
    value = document.get("fingerprints")
    return value if isinstance(value, list) else None


def compare_records(records, *, rtol=DEFAULT_RTOL) -> dict:
    """Group one suite's rows by case and compare every arm pair, output by output."""
    cases = defaultdict(dict)
    for record in records:
        fingerprints = fingerprints_of(record)
        # An arm that records no invariants (a scalar network result, or a
        # collector that does not emit them) takes part in no comparison.
        if not fingerprints:
            continue
        key = (record.get("suite_id"), record.get("problem_id"))
        cases[key][record.get("backend")] = fingerprints
    disagreements = []
    compared = 0
    unrecorded = []
    for (suite_id, problem_id), arms in sorted(cases.items()):
        names = sorted(arms)
        for index in range(len(names)):
            for other in names[index + 1:]:
                left, right = arms[names[index]], arms[other]
                if len(left) != len(right):
                    disagreements.append({
                        "suite_id": suite_id, "problem_id": problem_id,
                        "arms": [names[index], other], "output": None,
                        "problems": [f"{len(left)} outputs vs {len(right)}"],
                    })
                    continue
                for output, (a, b) in enumerate(zip(left, right)):
                    compared += 1
                    problems = compare_outputs(a, b, rtol=rtol)
                    if problems:
                        disagreements.append({
                            "suite_id": suite_id, "problem_id": problem_id,
                            "arms": [names[index], other], "output": output,
                            "problems": problems,
                        })
        if len(arms) < 2:
            unrecorded.append({"suite_id": suite_id, "problem_id": problem_id,
                               "arms": sorted(arms)})
    return {
        "compared_output_pairs": compared,
        "disagreements": disagreements,
        "single_arm_cases": unrecorded,
        "rtol": rtol,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", help="records.jsonl written by one suite")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--fail-on-disagreement", action="store_true",
                        help="exit 1 when two arms disagree (default: report only)")
    args = parser.parse_args(argv)
    records = []
    with open(args.records, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    report = compare_records(records, rtol=args.rtol)
    print(json.dumps(report, indent=1))
    if report["disagreements"]:
        print(
            f"{len(report['disagreements'])} arm disagreement(s) across "
            f"{report['compared_output_pairs']} compared output pairs",
            file=sys.stderr,
        )
    return 1 if (args.fail_on_disagreement and report["disagreements"]) else 0


if __name__ == "__main__":
    sys.exit(main())
