#!/usr/bin/env python3
"""Validate a complete public API matrix and publish medians/IQRs in microseconds."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

import gpu_bandwidth_guard as guard

GIGABYTE = 1e9


def compile_note(accounting):
    """One sentence naming this run's compile, from the totals the run recorded.

    A Rust arm cannot time NVRTC alone, so the run publishes the module count and the
    untimed wall of the cases that compiled, never a compile interval. Nothing here is
    taken from another run: a page describes the run that produced it.
    """
    modules = accounting["modules_compiled"]
    if modules:
        note = (f'This run compiled **{modules} NVRTC modules** in '
                f'{accounting["compiling_processes"]} of its arms, because the CUDA '
                f'artifact cache started cold.')
        wall = accounting.get("untimed_wall_seconds")
        if isinstance(wall, (int, float)):
            note += (f' The untimed wall of the cases that compiled is **{wall:.1f} s** of its '
                     f'{accounting["case_wall_seconds"]:.1f} s of recorded case wall clock, and '
                     f'that wall also holds their fixture read, transfers and verification, so '
                     f'it is not a compile time.')
        return note
    return ('This run did not compile: no NVRTC module was built in any of its arms. That '
            'describes this run only, and does not show that the CUDA artifact cache was warm.')


def load_rows(suite, paths, *, verify_only=False, allow_partial=False):
    """The rows this page needs.

    With `allow_partial`, a case an arm never reached is carried as a `missing`
    row and a failed row is published as a failure instead of refusing the whole
    page. The run still exits non-zero; this only decides whether the rows that
    did run get a report.
    """
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
            elif allow_partial:
                reason = (row.get("reason") or "").strip()
                row["reason"] = reason or f"{row['status']}: no reason recorded"
            else:
                raise ValueError(f"cannot publish failed measurement {key}: {row.get('reason')}; raw run retained")
            rows[key] = row
    if rows.keys() != wanted:
        if not allow_partial:
            raise ValueError(f"missing results: {sorted(wanted - rows.keys())}")
        for key in sorted(wanted - rows.keys()):
            rows[key] = {"case_id": key[0], "backend": key[1], "status": "missing",
                         "reason": "missing: the arm recorded no row for this case"}
    return rows


BATCHING_MARK = "†"


def mixed_batching_cases(suite, rows):
    """Case ids whose measured arms timed a different number of operations per sample.

    The per-operation median divides each batch by its own count, but a batch of 32 and
    a batch of 16 amortise launch and synchronization differently, so the two arms'
    values are an approximate comparison, not a like-for-like one (#336).
    """
    mixed = set()
    for case in suite["cases"]:
        counts = {rows[case["id"], backend]["timing"]["operations_per_sample"]
                  for backend in suite["backends"]
                  if rows[case["id"], backend]["status"] == "ok"}
        if len(counts) > 1:
            mixed.add(case["id"])
    return mixed


def render(suite, paths, metadata, timestamp, run_path=None, allow_partial=False):
    rows = load_rows(suite, paths, allow_partial=allow_partial)
    mixed = mixed_batching_cases(suite, rows)
    calibration = guard.calibration_of(metadata)
    for row in rows.values():
        if row["status"] != "ok":
            row.pop("samples_us", None)
    backends = suite["backends"]
    measured = sum(row["status"] == "ok" for row in rows.values())
    excluded_verified = sum(row["status"] == "unsupported" and bool(row.get("cuda_implementation_verified")) for row in rows.values())
    not_measured = sorted((key, row) for key, row in rows.items()
                          if row["status"] not in {"ok", "unsupported"})
    # The declared run is the one the page names, including a run collected under
    # an earlier suite-directory layout.
    raw_run = str(run_path or f'data/results/nvidia-gpu/{suite["suite_id"]}/{timestamp}').rstrip("/")
    if not suite.get("description"):
        raise ValueError("suite selection records no 'description'; the page note "
                         "that says what its shape class means cannot be written")
    text = [f'# {suite["suite_id"]}: CUDA public API benchmark', "",
            f'Measured cells: **{measured}**. Unsupported cells: **{len(rows)-measured}**, including **{excluded_verified}** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.', ""]
    if calibration:
        text += [f'Logical traffic: this run measured a plain device copy of {calibration["buffer_bytes"]} bytes at **{calibration["bytes_per_second"] / GIGABYTE:.1f} GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case\'s implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.', ""]
    else:
        text += ['This run metadata records no device copy rate, so the records carry the logical bytes without a copy ratio.', ""]
    text += [f'Raw run: `{raw_run}/`.', "",
            " ".join(str(suite["description"]).split()), ""]
    accounting = (metadata.get("collection") or {}).get("compile")
    if accounting:
        # This run's own compile accounting: the module count and the untimed wall of the
        # cases that compiled. No other run's figures are quoted here.
        text += [compile_note(accounting), ""]
    text += [
            'Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.', "",
            f'Cases marked {BATCHING_MARK} (**{len(mixed)}** of {len(suite["cases"])}) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.', "",
            'Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.', "",
            'All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.', "",
            '| Case | ' + ' | '.join(backends) + ' |',
            '| --- | ' + ' | '.join(['---:'] * len(backends)) + ' |']
    if any(case.get("dtype") in {"i32", "i64"} for case in suite["cases"]):
        text[-2:-2] = [
            'I32/I64 outputs require exact integer equality without conversion to floating point. Division and remainder truncate toward zero; axis-1 sum preserves the input dtype. Arithmetic fixtures avoid overflow, and I64 unary/order fixtures include values above 2**53 to detect loss of integer precision. A zero verification value in these rows means exact equality.', ""]
    if any(case.get("op") == "reshape" for case in suite["cases"]):
        text[-2:-2] = [
            'Reshape-copy cases add a leading singleton dimension and materialize the full output. Direct Tensor.reshape and EagerSession.reshape copy device data; PyTorch calls public view_copy. Direct/PyTorch initial/final allocation-pointer checks are untimed and reject input aliasing. Eager reshape copies its input before returning a descriptor view; original CUDA placement/shape/dtype are checked and a public inspection copy is made outside timing. JAX reshape has no established matching owned-output guarantee and remains unsupported; reshape-plus-copy composition is not substituted.', ""]
    if any(case.get("op") == "stack" for case in suite["cases"]):
        text[-2:-2] = [
            'Stack selects actual public Tensor.stack/EagerSession.stack with two distinct inputs and axis 0 or 2. Their intrinsic intermediate allocations are part of this public stack operation. PyTorch/JAX use native stack; final outputs are fully verified. Direct Rust initial/final pointer checks reject aliasing either input outside timing; eager original descriptors are checked before any untimed inspection copy.', ""]
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
            elif row["status"] != "ok":
                cells.append(row["status"])
            else:
                v = row["samples_us"]
                cells.append(f'{np.median(v):.3f} ({np.percentile(v, 75)-np.percentile(v, 25):.3f})')
        label = case["id"] + (f" {BATCHING_MARK}" if case["id"] in mixed else "")
        text.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    text += ["", "## Sampling", "", "| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    # Sorted, so the page does not depend on the order the backends' files were
    # passed in (the collector starts Rust arms before Python arms).
    for key, row in sorted(rows.items()):
        traffic = (row.get("traffic") or {}).get("bytes_per_operation")
        traffic_cell = str(traffic) if isinstance(traffic, int) else "unknown"
        ratio = (row.get("bandwidth_guard") or {}).get("copy_rate_ratio")
        ratio_cell = f"{ratio:.3f}" if isinstance(ratio, (int, float)) else "unavailable"
        if row["status"] == "ok":
            t = row["timing"]
            text.append(f'| {key[0]} | {key[1]} | {t["operations_per_sample"]} | {len(t["batch_ms"])} | {np.median(t["batch_ms"]):.3f} | {traffic_cell} | {ratio_cell} | {row["verification"]["max_error_over_tolerance"]:.4g} |')
        else:
            text.append(f'| {key[0]} | {key[1]} | — | — | — | {traffic_cell} | {ratio_cell} | {row["reason"].replace("|", "/")} |')
    if not_measured:
        text += ["", "## Not measured", "",
                 "These cells carry no number: the arm failed or never recorded the case. "
                 "The run still reports failure; the rows that did run are above.", "",
                 "| Case | Backend | Status | Reason |", "| --- | --- | --- | --- |"]
        for (case, backend), row in not_measured:
            text.append(f'| {case} | {backend} | {row["status"]} | '
                        f'{str(row.get("reason") or "").replace("|", "/")} |')
    text += ["", "## Run metadata", "", "```yaml", yaml.safe_dump(metadata, sort_keys=False).rstrip(), "```", ""]
    return "\n".join(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--run-metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--allow-partial", action="store_true",
                   help="render the rows that ran when another arm failed or missed cases")
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
        content = render(suite, args.inputs, metadata, args.run_metadata.parent.name,
                         allow_partial=args.allow_partial)
        content += overhead_warning_section(load_rows(suite, args.inputs,
                                                      allow_partial=args.allow_partial))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    temp.write_text(content)
    temp.replace(args.output)


def overhead_warning_section(rows):
    """The recorded untimed-budget warnings as a page section, or "".

    The budget has no measured basis, so a case over it is reported here and is still
    published with its number: the section lists what was recorded and claims nothing.
    """
    warned = [(key, warning) for key, row in rows.items()
              if (warning := (row.get("overhead") or {}).get("warning"))]
    if not warned:
        return ""
    budget = (rows[warned[0][0]].get("overhead") or {}).get("budget_ratio")
    text = ["", "## Untimed work over the recorded budget", "",
            f'Each row records the wall clock it spent outside its timed region. These cases '
            f'spent more than the recorded budget ratio ({budget}) there. The threshold has no '
            'measured basis, so this is a recorded warning: it does not change a cell, its '
            'acceptance or the exit status of the run.', "",
            "| Case | Backend | Warning |", "| --- | --- | --- |"]
    for key, warning in warned:
        text.append(f'| {key[0]} | {key[1]} | {warning.replace("|", "/")} |')
    return "\n".join(text)


if __name__ == "__main__":
    main()
