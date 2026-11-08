#!/usr/bin/env python3
"""Check downloaded Rust AD outputs before allowing report publication.

Which check runs is the suite problem's ``verify.mode``:

* ``cpu_reference`` re-executes the loss through ``torch.func`` on the CPU and
  compares every element -- the original gate, and the default.
* ``analytic`` compares every element against a closed-form derivative built on
  the accelerator, so a GPU-sized case is checked without a second AD pass.
* ``finite_difference`` compares the row's directional derivative with a central
  difference of the primal loss, at a step derived from f64 precision.
* ``measure_only`` runs nothing and keeps the manifest's reason on the record.

Every mode writes the same verification fields, so one reader covers all three
and the dashboard can say which check backed a number.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import yaml

import linalg_ad_verify_modes as modes
from benchmark_cpu_ops_python import data, matrix_for_linalg_ad, tangent_seed


def _inputs(problem):
    ad = problem["linalg_ad"]
    n, seed, loss = ad["n"], ad["matrix_seed"], ad["loss"]
    matrix = matrix_for_linalg_ad(loss, n, seed)
    tangent = data((n, n), tangent_seed(seed))
    rhs = (
        data((n, ad.get("rhs_cols", 1)), ad.get("rhs_seed", 0))
        if loss == "grad_sum_solve"
        else None
    )
    return matrix, tangent, rhs


def _outputs(row, phase):
    notes = json.loads(row["execution"]["notes"])
    actual = []
    for value in notes["verification_outputs"][phase]:
        if value["dtype"] != "f64":
            raise ValueError("Rust AD output dtype mismatch")
        actual.append(
            np.asarray(value["values_col_major"], dtype=np.float64).reshape(
                value["shape"], order="F"
            )
        )
    return actual


def validate(row, problem, defaults=None):
    if row["status"] not in {"ok", "verification_only", "unsupported"}:
        return row
    # What this row costs to prove: CPU AD reference plus central difference, then
    # two elementwise comparisons. It is recorded so the suite's verification pass
    # does not have to be guessed from the total process time.
    started = time.perf_counter()
    try:
        notes = json.loads(row["execution"].get("notes") or "{}")
    except json.JSONDecodeError:
        # Failed unsupported hooks have no output artifact. Their original
        # explanatory notes are prose, and never count as verified execution.
        if row["status"] == "unsupported":
            return row
        raise
    mode = modes.mode_for(problem, defaults)
    if mode == "measure_only":
        # No check runs, and no reference is built. The row must already state
        # the opt-out: a measured row that still claims `skipped` or `passed`
        # was not produced under this mode, so it is not published as one.
        if row["status"] == "ok" and row["verification"].get("status") != "not_run":
            raise ValueError(
                "measure_only row records verification.status="
                f"{row['verification'].get('status')!r}, expected 'not_run'"
            )
        return row
    outputs = notes.pop("verification_outputs", None)
    if outputs is None:
        if row["status"] in {"ok", "verification_only"}:
            raise ValueError("Rust AD row has no numerical outputs")
        return row
    loss = problem["linalg_ad"]["loss"]
    matrix, tangent, rhs = _inputs(problem)
    rtol, atol = modes.tolerances(problem, defaults)
    verdict = None
    for phase in ("initial", "final"):
        candidate = modes.check(
            mode,
            loss,
            problem["op"],
            _outputs(row, phase),
            matrix,
            tangent,
            rhs,
            rtol,
            atol,
        )
        verdict = (
            candidate
            if verdict is None
            else {
                **candidate,
                "max_abs_error": max(verdict["max_abs_error"], candidate["max_abs_error"]),
                "max_rel_error": max(verdict["max_rel_error"], candidate["max_rel_error"]),
                "residual": max(verdict["residual"], candidate["residual"]),
            }
        )
    row["verification"].update(
        status="passed",
        mode=mode,
        reference_backend=verdict["reference_backend"],
        max_abs_error=verdict["max_abs_error"],
        max_rel_error=verdict["max_rel_error"],
        residual=verdict["residual"],
        rtol=rtol,
        atol=atol,
        reason=verdict["reason"],
    )
    notes.update(
        verification_mode=mode,
        verification_diagnostic=verdict["diagnostic"],
        cuda_implementation_verified=True,
        numerical_output_artifact="raw Rust records retain initial/final downloaded outputs",
    )
    row["execution"]["notes"] = json.dumps(notes)
    row.setdefault("timing", {})["verification_time_ms"] = round(
        (time.perf_counter() - started) * 1000, 3)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("suite", type=Path)
    args = parser.parse_args()
    suite = yaml.safe_load(args.suite.read_text())
    defaults = suite.get("defaults")
    problems = {p["id"]: p for p in suite["problems"]}
    rows = []
    for line in args.input.read_text().splitlines():
        row = json.loads(line)
        problem = problems.get(row["problem_id"], {})
        try:
            rows.append(validate(row, problem, defaults))
        except modes.VerificationFailure as exc:
            row["status"] = "verification_failed"
            row["verification"].update(
                status="failed", mode=exc.verdict["mode"], reason=str(exc),
                max_abs_error=exc.verdict["max_abs_error"],
                max_rel_error=exc.verdict["max_rel_error"],
                residual=exc.verdict["residual"],
            )
            rows.append(row)
        except Exception as exc:
            row["status"] = "verification_failed"
            row["verification"].update(status="failed", reason=str(exc))
            rows.append(row)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in rows))
    if any(r["status"] in {"verification_failed", "runtime_failed"} for r in rows):
        raise SystemExit(
            "Rust AD verification/execution failed; raw data retained, report not published"
        )


if __name__ == "__main__":
    main()
