#!/usr/bin/env python3
"""Check downloaded Rust AD outputs before allowing report publication."""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml

from benchmark_gpu_linalg_ad import LOSS_TORCH
from benchmark_cpu_ops_python import data, matrix_for_linalg_ad, tangent_seed
from gpu_linalg_ad_contracts import cpu_reference, verify_arrays


def validate(row, problem):
    notes = json.loads(row["execution"].get("notes") or "{}")
    outputs = notes.pop("verification_outputs", None)
    if outputs is None:
        if row["status"] in {"ok", "verification_only"}:
            raise ValueError("Rust AD row has no numerical outputs")
        return row
    ad = problem["linalg_ad"]
    n, seed, loss = ad["n"], ad["matrix_seed"], ad["loss"]
    matrix = matrix_for_linalg_ad(loss, n, seed)
    tangent = data((n, n), tangent_seed(seed))
    rhs = data((n, ad.get("rhs_cols", 1)), ad.get("rhs_seed", 0)) if loss == "grad_sum_solve" else None
    expected, difference = cpu_reference(LOSS_TORCH[loss], matrix, tangent, rhs, problem["op"])
    rtol, atol = problem.get("verify", {}).get("rtol", 1e-5), problem.get("verify", {}).get("atol", 1e-8)
    errors = []
    for phase in ("initial", "final"):
        actual = []
        for value in outputs[phase]:
            if value["dtype"] != "f64":
                raise ValueError("Rust AD output dtype mismatch")
            actual.append(np.asarray(value["values_col_major"], dtype=np.float64).reshape(value["shape"], order="F"))
        errors.append(verify_arrays(actual, expected, rtol, atol))
    row["verification"].update(status="passed", reference_backend="pytorch-cpu-f64 AD plus central difference of primal loss",
        max_abs_error=max(e[0] for e in errors), max_rel_error=max(e[1] for e in errors),
        residual=max(e[2] for e in errors), rtol=rtol, atol=atol, reason=None)
    notes.update(central_difference=difference, cuda_implementation_verified=True,
                 numerical_output_artifact="raw Rust records retain initial/final downloaded outputs")
    row["execution"]["notes"] = json.dumps(notes)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("suite", type=Path)
    args = parser.parse_args()
    suite = yaml.safe_load(args.suite.read_text())
    problems = {p["id"]: p for p in suite["problems"]}
    rows = []
    for line in args.input.read_text().splitlines():
        row = json.loads(line)
        try:
            rows.append(validate(row, problems[row["problem_id"]]))
        except Exception as exc:
            row["status"] = "verification_failed"
            row["verification"].update(status="failed", reason=str(exc))
            rows.append(row)
    args.output.write_text("".join(json.dumps(r)+"\n" for r in rows))
    if any(r["status"] in {"verification_failed", "runtime_failed"} for r in rows):
        raise SystemExit("Rust AD verification/execution failed; raw data retained, report not published")


if __name__ == "__main__":
    main()
