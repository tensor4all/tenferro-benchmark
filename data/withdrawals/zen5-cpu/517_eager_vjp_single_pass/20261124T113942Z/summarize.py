#!/usr/bin/env python3
"""Per (benchmark, shape, arm): median of process medians, between-start SD and W.

W is the median over process starts of the within-process sample SD, the same
quantity campaign_toolkit's report_run prints as `W`. Run from the repository root:

    python3 data/withdrawals/zen5-cpu/517_eager_vjp_single_pass/20261124T113942Z/summarize.py
"""
import collections
import json
import pathlib
import statistics as st

HERE = pathlib.Path(__file__).resolve().parent
cells = collections.defaultdict(list)
for path in sorted(HERE.glob("raw_*.jsonl")):
    arm = path.name.split("_")[1]
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("record_type") != "samples":
            continue
        samples = [ns / 1e6 for ns in record["per_operation_ns"]]
        cells[(record["benchmark"], record["shape"], arm)].append(
            (st.median(samples), st.stdev(samples), record["eager_session_policy"]))
print("| benchmark | shape | arm | starts | median of medians (ms) | between-start SD | W | policy |")
print("|---|---|---|---:|---:|---:|---:|---|")
for (benchmark, shape, arm), starts in sorted(cells.items()):
    medians = [m for m, _, _ in starts]
    print(f"| {benchmark} | {shape} | {arm} | {len(starts)} | {st.median(medians):.3f} | "
          f"{st.stdev(medians):.3f} | {st.median([s for _, s, _ in starts]):.3f} | {starts[0][2]} |")
