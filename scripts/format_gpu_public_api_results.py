#!/usr/bin/env python3
"""Validate a complete public API matrix and publish medians/IQRs in microseconds."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


def load_rows(suite, paths, *, verify_only=False):
    wanted = {(c["id"], b) for c in suite["cases"] for b in suite["backends"]}
    rows = {}
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = row["case_id"], row["backend"]
            if key in rows or key not in wanted:
                raise ValueError(f"duplicate or unselected result: {key}")
            if row["status"] == ("verification_only" if verify_only else "ok"):
                error = row["verification"]["max_error_over_tolerance"]
                if not math.isfinite(error) or error > 1 or error < 0:
                    raise ValueError(f"invalid verification: {key}")
                if verify_only:
                    if row.get("timing") is not None:
                        raise ValueError(f"timing present in verification-only result: {key}")
                else:
                    timing = row["timing"]
                    count = timing["operations_per_sample"]
                    samples = timing["batch_ms"]
                    if not isinstance(count, int) or count < 1 or not samples or any(not math.isfinite(v) or v <= 0 for v in samples):
                        raise ValueError(f"invalid sampling: {key}")
                    row["samples_us"] = np.asarray(samples) * 1000 / count
            elif row["status"] == "unsupported":
                if not row.get("reason") or row.get("timing") is not None:
                    raise ValueError(f"invalid unsupported disposition: {key}")
                if row.get("cuda_implementation_verified"):
                    error = row["verification"]["max_error_over_tolerance"]
                    if not math.isfinite(error) or not 0 <= error <= 1:
                        raise ValueError(f"invalid excluded verification: {key}")
            else:
                raise ValueError(f"cannot publish failed measurement {key}: {row.get('reason')}; raw run retained")
            rows[key] = row
    if rows.keys() != wanted:
        raise ValueError(f"missing results: {sorted(wanted - rows.keys())}")
    return rows


def render(suite, paths, metadata, timestamp):
    rows = load_rows(suite, paths)
    backends = suite["backends"]
    measured = sum(row["status"] == "ok" for row in rows.values())
    excluded_verified = sum(row["status"] == "unsupported" and bool(row.get("cuda_implementation_verified")) for row in rows.values())
    text = [f'# {suite["suite_id"]}: CUDA public API benchmark', "",
            f'Measured cells: **{measured}**. Unsupported cells: **{len(rows)-measured}**, including **{excluded_verified}** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.', "",
            f'Raw run: `data/results/nvidia-gpu/{suite["suite_id"]}/{timestamp}/`.', "",
            'Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.', "",
            'Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.', "",
            'All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.', "",
            '| Case | ' + ' | '.join(backends) + ' |',
            '| --- | ' + ' | '.join(['---:'] * len(backends)) + ' |']
    if any(case.get("dtype") in {"i32", "i64"} for case in suite["cases"]):
        text[-2:-2] = [
            'I32/I64 outputs require exact integer equality without conversion to floating point. Division and remainder truncate toward zero; axis-1 sum preserves the input dtype. Arithmetic fixtures avoid overflow, and I64 unary/order fixtures include values above 2**53 to detect loss of integer precision. A zero verification value in these rows means exact equality.', ""]
    if any(case.get("read_op") for case in suite["cases"]):
        text[-2:-2] = [
            'Transposed borrowed-read cases use prepared native CUDA views in tenferro and PyTorch. JAX compares the same logical values in its native representation; its columns do not claim an equivalent public strided-view contract. View creation and input conversion are outside timing.', ""]
    if any(case.get("jax_ref_addupdate") for case in suite["cases"]):
        text[-2:-2] = [
            'Unit-coefficient AXPBY uses native JAX Ref.addupdate with caller-owned fixed buffers. Its synchronization includes one GPU scalar completion token per destination and block_until_ready; tokens are retained through clock stop and included in the logical retention bound. Helper compilation and seed restoration are untimed. Full Ref snapshots are used only for untimed inspection. This scalar-token synchronization cost is included in the reported latency; it performs no full output copy or D2H transfer inside timing.', ""]
    if any(case.get("op") == "dot_general_axes" and case.get("variant") == "batch" for case in suite["cases"]):
        text[-2:-2] = [
            'Generalized batch dot preserves native output order: tenferro [m,n,b], PyTorch/JAX [b,m,n]. Input permutation views and dimension-number configurations are prepared before timing. Output axes are normalized only on downloaded CPU arrays for verification; no timed output transpose or copy is added.', ""]
    if any(case.get("op", "").startswith("einsum_plan") for case in suite["cases"]):
        text[-2:-2] = [
            'Prepared einsum uses the public ConcreteEinsumPlan for ik,kj->ij, built once outside execution and sampling. Native matrix multiplication/reuse/accumulation references match the selected mathematical and output-ownership contracts; they do not assert an equivalent prepared-plan interface. Transposed input views/values, read/write handles and accumulation seed restoration are untimed.', ""]
    for case in suite["cases"]:
        cells = []
        for backend in backends:
            row = rows[case["id"], backend]
            if row["status"] == "unsupported":
                cells.append("unsupported")
            else:
                v = row["samples_us"]
                cells.append(f'{np.median(v):.3f} ({np.percentile(v, 75)-np.percentile(v, 25):.3f})')
        text.append('| ' + case["id"] + ' | ' + ' | '.join(cells) + ' |')
    text += ["", "## Sampling", "", "| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for key, row in rows.items():
        if row["status"] == "ok":
            t = row["timing"]
            text.append(f'| {key[0]} | {key[1]} | {t["operations_per_sample"]} | {len(t["batch_ms"])} | {np.median(t["batch_ms"]):.3f} | {row["verification"]["max_error_over_tolerance"]:.4g} |')
        else:
            text.append(f'| {key[0]} | {key[1]} | — | — | — | {row["reason"].replace("|", "/")} |')
    text += ["", "## Run metadata", "", "```yaml", yaml.safe_dump(metadata, sort_keys=False).rstrip(), "```", ""]
    return "\n".join(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--run-metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("inputs", nargs="+", type=Path)
    args = p.parse_args()
    suite = json.loads(args.suite.read_text())
    metadata = yaml.safe_load(args.run_metadata.read_text())
    if args.verify_only:
        rows = load_rows(suite, args.inputs, verify_only=True)
        summary = {"suite_id": suite["suite_id"], "gpu_assignment": metadata["gpu_assignment"],
                   "verified_rows": sum(r["status"] == "verification_only" for r in rows.values()),
                   "unsupported_rows": sum(r["status"] == "unsupported" for r in rows.values()),
                   "numerically_verified_excluded_rows": sum(r.get("cuda_implementation_verified", False) for r in rows.values())}
        content = json.dumps(summary, indent=2) + "\n"
        print(content, end="")
    else:
        content = render(suite, args.inputs, metadata, args.run_metadata.parent.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    temp.write_text(content)
    temp.replace(args.output)


if __name__ == "__main__":
    main()
