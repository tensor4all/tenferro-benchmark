#!/usr/bin/env python3
"""Paired warmed operation Ir, never native time or predicted speedup."""
import gzip
import json
import re
import statistics
from pathlib import Path

root = Path(__file__).parent

def read_text(path):
    if path.exists():
        return path.read_text()
    with gzip.open(str(path) + ".gz", "rt") as stream:
        return stream.read()

results = {}
for case, repeats, divisor, threshold in (
    ("bin_batched_matmul_b32_m128_n128_k128", (1, 2, 3), 2, 5),
    ("bin_matmul_1024", (1, 2, 3), 2, -1),
    ("lm_batch_likelihood_sentence_3_12d", (1,), 4, None),
):
    samples = []
    for repeat in repeats:
        row = {}
        for variant in ("baseline", "final"):
            counts = []
            kernels = []
            for runs in (1, 3):
                prefix = root / "raw/final" / f"r{repeat}-{variant}-{case}-n{runs}"
                text = read_text(prefix.with_suffix(".callgrind"))
                assert re.search(r"^events: Ir$", text, re.M)
                counts.append(int(re.search(r"^summary: (\d+)$", text, re.M)[1]))
                annotation = read_text(Path(f"{prefix}-annotated.txt"))
                kernels.append(sum(int(re.match(r"\s*([\d,]+)", line)[1].replace(",", ""))
                                   for line in annotation.splitlines()
                                   if "dgemm_kernel_HASWELL" in line))
            row[variant] = (counts[1] - counts[0]) / divisor
            row[f"{variant}_gemm_kernel_self"] = (kernels[1] - kernels[0]) / divisor
        assert row["baseline_gemm_kernel_self"] == row["final_gemm_kernel_self"]
        row["reduction_percent"] = 100 * (1 - row["final"] / row["baseline"])
        samples.append(row)
    results[case] = {
        "samples": samples,
        "baseline_median_Ir": statistics.median(s["baseline"] for s in samples),
        "final_median_Ir": statistics.median(s["final"] for s in samples),
        "paired_reduction_percent_range": [min(s["reduction_percent"] for s in samples), max(s["reduction_percent"] for s in samples)],
        "all_pairs_pass_instruction_gate": None if threshold is None else all(s["reduction_percent"] >= threshold for s in samples),
    }
output = {"baseline_tenferro": "03ec9804beca51dce6355d1751e10251ff2972d4", "candidate_tenferro": "57bee76de1a752782bdd985f332050422f885372", "strided": "17e05ffb168d0f826529ea2c3aceeb4ec851448b", "provider": "OpenBLAS 0.3.26", "threads": 1, "cpu": 16, "metric": "whole-process N3-N1 Ir; divide by 2 for cached identical binary paths, by 4 for the two distinct LM paths; three pairs for binary controls, one exploratory LM pair", "native_speedup": None, "cases": results}
(root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
