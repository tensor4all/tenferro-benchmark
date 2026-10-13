#!/usr/bin/env python3
"""Render GPU0 whole-sequence latency and separate GPU1 untimed counters."""
import argparse
import json
import math
from pathlib import Path
import statistics

import yaml


def read_run(path):
    metadata = yaml.safe_load((path / "run.yaml").read_text())
    provenance = json.loads((path / "provenance.json").read_text())
    rows = {}
    for line in (path / "samples_t1.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = row["case_id"]
        if key in rows:
            raise ValueError(f"duplicate case: {key}")
        if row["status"] != "passed" or row["correctness_status"] != "passed":
            raise ValueError(f"unsuccessful case: {key}")
        error = row["max_rel_error"]
        if not math.isfinite(error) or not 0 <= error <= 1e-10:
            raise ValueError(f"invalid numerical verification: {key}")
        if row["kind"] != "shape_stream" or row["dtype"] != "f64" or row["threads"] != 1:
            raise ValueError(f"unexpected operation/dtype/threads: {key}")
        if row["operations_per_sequence"] != len(row["sequence"]):
            raise ValueError(f"sequence length mismatch: {key}")
        if not 0 < row["retained_output_bytes"] <= 256 << 20:
            raise ValueError(f"invalid retention bound: {key}")
        rows[key] = row
    if not metadata["harness"]["commit"].startswith(provenance["harness_commit"]):
        raise ValueError("source revision mismatch")
    if metadata["cuda"]["device_uuid"] != provenance["device_uuid"]:
        raise ValueError("GPU provenance mismatch")
    return metadata, provenance, rows


def stats_text(values):
    median = statistics.median(values) / 1000
    q = statistics.quantiles(values, n=4, method="inclusive")
    return f"{median:.2f} ({(q[2] - q[0]) / 1000:.2f})"


def render(cases, timing_path, counter_path):
    tm, tp, timed = read_run(timing_path)
    cm, cp, checked = read_run(counter_path)
    selected = [c for c in cases if c["kind"] == "shape_stream"]
    latency_ids = [c["id"] for c in selected if not c["params"]["instrument"]]
    counter_ids = [c["id"] for c in selected if c["params"]["instrument"]]
    if set(timed) != set(latency_ids) or not set(counter_ids) <= checked.keys():
        raise ValueError("missing or unexpected shape-stream cases")
    if set(checked) - {c["id"] for c in selected}:
        raise ValueError("unexpected verification cases")
    for case in selected:
        for rows in (timed, checked):
            if case["id"] in rows:
                row = rows[case["id"]]
                if (row["params"] != case["params"] or row["measurement_scope"] != case["measurement_scope"]
                        or row["plan_cache_capacity"] != 64):
                    raise ValueError(f"case contract mismatch: {case['id']}")
    for case in (c for c in selected if c["params"]["instrument"]):
        match = next(c["id"] for c in selected if not c["params"]["instrument"]
                     and c["params"]["n"] == case["params"]["n"] and c["arm"] == case["arm"])
        if checked[case["id"]]["sequence"] != timed[match]["sequence"]:
            raise ValueError("timing/counter sequences differ")
    for provenance, role, device in ((tp, "benchmark", 0), (cp, "debug", 1)):
        if provenance["gpu_role"] != role or provenance["physical_device_index"] != device:
            raise ValueError("incorrect GPU assignment")
    if not cp.get("verification_only") or tp["effort"] != "scan":
        raise ValueError("expected scan timings and a verification-only counter pass")
    if tm["tenferro_rs"]["commit"] != cm["tenferro_rs"]["commit"]:
        raise ValueError("tenferro revisions differ")
    if tp["shape_stream_source_sha256"] != cp["source_sha256"]:
        raise ValueError("shape-stream operation source differs between passes")
    if any(row["samples"] for row in checked.values()):
        raise ValueError("timings present in the untimed counter/verification pass")

    lines = ["# CUDA whole-sequence shape/cache diagnostics", "",
             f"- Timings: `{timing_path.as_posix()}` (physical GPU0, `{tp['device_uuid']}`).",
             f"- Counters/checks: `{counter_path.as_posix()}` (physical GPU1, `{cp['device_uuid']}`).",
             f"- Hardware: {tm['cuda']['device_name']}; driver {tm['cuda']['driver_version']}.",
             f"- tenferro-rs: `{tm['tenferro_rs']['commit']}`.",
             f"- Harness: timings `{tm['harness']['commit']}`, counters `{cm['harness']['commit']}`; operation source hashes match.", "",
             f"Selected family: {len(latency_ids)} latency cases and {len(counter_ids)} separate counter cases. "
             "Other gpu/perf_issues cases are unselected here; this report establishes no full-suite coverage.", "",
             f"Effort: scan, {tp['warmups']} warmup(s), {tp['samples']} measured sequences per case. "
             "CoV > 10% is a descriptive NOISY label. No paired baseline/candidate performance conclusion is drawn.", "",
             "Each sample times an entire F64 batched-contraction sequence, including required planning, "
             "then native CUDA synchronization. Host enqueue is the interval before that final synchronization; "
             "neither column is device-kernel-only time. Inputs/uploads, entered session, library initialization, "
             "cache reset, declared key prewarming and retention storage are outside timing. "
             "Full-output validation/download/destruction follow clock stop. The plan bound is 64. "
             "CubeCL compiled kernels, cuTENSOR handle and allocator pools remain initialized.", "",
             "## Sequence latency", "",
             "Values are median (inclusive IQR) µs per **sequence**. Sequence lengths and shapes differ by mode; "
             "the table reports each declared workflow without converting their differences into operation-speed ratios.", "",
             "| Case | Calls/sequence | Host enqueue µs | Synchronized µs | CoV | Status |",
             "|---|---:|---:|---:|---:|---|"]
    for key in latency_ids:
        row = timed[key]
        samples = row["samples"]
        enqueue = row["host_enqueue_samples_ns"]
        if row["instrumentation_only"] or row["counter_observations"]:
            raise ValueError(f"instrumentation in a timing pass: {key}")
        if len(samples) != tp["samples"] or len(samples) < 3 or len(enqueue) != len(samples):
            raise ValueError(f"invalid sample count: {key}")
        elapsed = []
        for i, (sample, queued) in enumerate(zip(samples, enqueue)):
            duration = sample["elapsed_ns"]
            if (sample["sample_index"] != i or sample["iterations"] != 1 or
                    not math.isfinite(duration) or not 0 < queued <= duration):
                raise ValueError(f"invalid sequence timing: {key}")
            elapsed.append(duration)
        cov = statistics.stdev(elapsed) / statistics.mean(elapsed)
        lines.append(f"| {key} | {row['operations_per_sequence']} | {stats_text(enqueue)} | "
                     f"{stats_text(elapsed)} | {cov:.1%} | {'NOISY' if cov > .10 else 'scan'} |")

    lines += ["", "## Untimed public counters on GPU1", "",
              "Deltas use the first snapshot after reset/prewarming and the final snapshot after output drop. "
              "Workspace bytes describe retained shared scratch, not total allocator reservations. "
              "All per-command enqueue/completion/drop observations remain in the raw records.", "",
              "| Case | Entries before→drop | Hits Δ | Misses Δ | Evictions Δ | Plan bytes at drop | Workspace bytes at drop | Observed enqueue in-flight peak |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in counter_ids:
        row = checked[key]
        observations = row["counter_observations"]
        phases = [o["phase"] for o in observations]
        count = row["operations_per_sequence"]
        if (not row["instrumentation_only"] or
                phases != ["before"] + ["enqueue"] * count + ["completion", "drop"]):
            raise ValueError(f"invalid counter phases: {key}")
        for i, observation in enumerate(observations[1:1 + count]):
            if observation["index"] != i or observation["key"] != row["sequence"][i]["key"]:
                raise ValueError(f"counter sequence mismatch: {key}")
        before = observations[0]["stats"]["plans"]
        final = observations[-1]["stats"]
        after = final["plans"]
        deltas = [after[k] - before[k] for k in ("hits", "misses", "evictions")]
        if any(v < 0 for v in deltas):
            raise ValueError(f"counter reset within pass: {key}")
        peak = max(o["stats"]["retirement"]["in_flight"] for o in observations[1:1 + count])
        lines.append(f"| {key} | {before['entries']}→{after['entries']} | " +
                     " | ".join(map(str, deltas)) +
                     f" | {after['retained_bytes']} | {final['workspace']['bytes']} | {peak} |")
    lines += ["", "## Exact sequence definitions", "",
              "Ordered keys and public input shape/stride descriptors used by each timing case. "
              "Repeated keys share the descriptor recorded once below; dtype is F64."]
    for key in latency_ids:
        sequence = timed[key]["sequence"]
        unique = {s["key"]: s for s in sequence}
        if any(unique[s["key"]] != s for s in sequence):
            raise ValueError(f"inconsistent repeated key: {key}")
        lines += ["", f"### {key}", "", "```json",
                  json.dumps({"order": [s["key"] for s in sequence], "descriptors": list(unique.values())}, separators=(",", ":")),
                  "```"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("data/instances/gpu_perf_issues.json"))
    parser.add_argument("--timing-run", type=Path, required=True)
    parser.add_argument("--counter-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = render(json.loads(args.cases.read_text()), args.timing_run, args.counter_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)


if __name__ == "__main__":
    main()
