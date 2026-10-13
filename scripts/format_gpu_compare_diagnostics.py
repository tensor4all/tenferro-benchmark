#!/usr/bin/env python3
"""Describe within-process compare drift; never publish operation coverage."""
import argparse
import json
import math
import statistics
from pathlib import Path

import yaml


def render(root):
    suite = json.loads((root / "suite.json").read_text())
    wanted = [case["id"] for case in suite["cases"]]
    metadata = yaml.safe_load((root / "run.yaml").read_text())
    text = ["# Small CUDA compare: lifetime diagnostic", "",
            "Diagnostic only: these nonstationary samples are not steady-state operation coverage or a regression/speedup claim. Physical GPU1 was used for debugging; no GPU0 publication timings are replaced.", "",
            f"Raw diagnostic: `{root}/`. Three independent processes ran the same case sequence, each with 3 warmups and 30 batched samples. Setup and validation are untimed; each batch retains outputs through native device synchronization and clock stop.", "",
            "Sequence: " + ", ".join(f"`{name}`" for name in wanted) + ".", "",
            "Each case creates a backend, but process-global device allocation state can survive between cases. First/last columns describe the first/last ten samples in that process; they are not balanced candidate/baseline comparisons.", "",
            "| Trial | Case | Ops/sample | Median µs/op | First ten median batch ms | Last ten median batch ms | Last/first |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for trial in range(1, 4):
        rows = [json.loads(line) for line in (root / f"trial_{trial}.jsonl").read_text().splitlines()]
        if [row["case_id"] for row in rows] != wanted:
            raise ValueError("missing, duplicate or reordered diagnostic cases")
        for row in rows:
            error = row["verification"]["max_error_over_tolerance"]
            if row["status"] != "ok" or not math.isfinite(error) or not 0 <= error <= 1:
                raise ValueError("failed diagnostic numerical validation")
            if row["gpu_role"] != "debug" or row["physical_device_index"] != 1:
                raise ValueError("wrong diagnostic GPU assignment")
            samples = row["timing"]["batch_ms"]
            count = row["timing"]["operations_per_sample"]
            if len(samples) != 30 or count < 1 or any(not math.isfinite(v) or v <= 0 for v in samples):
                raise ValueError("invalid diagnostic samples")
            first, last = statistics.median(samples[:10]), statistics.median(samples[-10:])
            text.append(f"| {trial} | {row['case_id']} | {count} | {statistics.median(samples)*1000/count:.3f} | {first:.3f} | {last:.3f} | {last/first:.3f} |")
    text += ["", "Source observation: at the recorded tenferro revision, `cubecl/dispatch.rs::alloc_bool_output` calls `client.empty` directly; numeric `alloc_output` first consults `take_output_handle`. This difference is a hypothesis for allocator/lifetime investigation, not an established cause. Cache/retirement instrumentation and controlled confirmations remain part of issue #107.", "",
             "Run metadata was collected immediately after the probes with the same checkout and GPU environment; it is not a per-phase timing record.", "", "```yaml", yaml.safe_dump(metadata, sort_keys=False).rstrip(), "```", ""]
    return "\n".join(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(args.root))
