#!/usr/bin/env python3
"""Validate a complete public API matrix and publish medians/IQRs in microseconds."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


def render(suite, paths, metadata, timestamp):
    wanted = {(c["id"], b) for c in suite["cases"] for b in suite["backends"]}
    rows = {}
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = row["case_id"], row["backend"]
            if key in rows or key not in wanted:
                raise ValueError(f"duplicate or unselected result: {key}")
            if row["status"] == "ok":
                timing = row["timing"]
                count = timing["operations_per_sample"]
                samples = timing["batch_ms"]
                if not isinstance(count, int) or count < 1 or not samples or any(not math.isfinite(v) or v <= 0 for v in samples):
                    raise ValueError(f"invalid sampling: {key}")
                error = row["verification"]["max_error_over_tolerance"]
                if not math.isfinite(error) or error > 1 or error < 0:
                    raise ValueError(f"invalid verification: {key}")
                row["samples_us"] = np.asarray(samples) * 1000 / count
            elif row["status"] != "unsupported":
                raise ValueError(f"cannot publish failed measurement {key}: {row.get('reason')}; raw run retained")
            rows[key] = row
    if rows.keys() != wanted:
        raise ValueError(f"missing results: {sorted(wanted - rows.keys())}")
    backends = suite["backends"]
    text = [f'# {suite["suite_id"]}: CUDA public API benchmark', "",
            f'Raw run: `data/results/nvidia-gpu/{suite["suite_id"]}/{timestamp}/`.', "",
            'Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.', "",
            'Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; `_into` rows instead reuse preallocated destinations across samples. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.', "",
            'All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.', "",
            '| Case | ' + ' | '.join(backends) + ' |',
            '| --- | ' + ' | '.join(['---:'] * len(backends)) + ' |']
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
    p.add_argument("inputs", nargs="+", type=Path)
    args = p.parse_args()
    content = render(json.loads(args.suite.read_text()), args.inputs,
                     yaml.safe_load(args.run_metadata.read_text()), args.run_metadata.parent.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    temp.write_text(content)
    temp.replace(args.output)


if __name__ == "__main__":
    main()
