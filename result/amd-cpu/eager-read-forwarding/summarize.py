#!/usr/bin/env python3
import gzip
import json
import re
import statistics
from pathlib import Path

root = Path(__file__).parent
protocol = json.loads((root / "protocol.json").read_text())

def text(path):
    if path.exists():
        return path.read_text()
    with gzip.open(str(path) + ".gz", "rt") as stream:
        return stream.read()

results = {}
for case in protocol["cases"]:
    samples = []
    for repeat in range(1, protocol["repetitions"] + 1):
        row = {}
        for variant in ("baseline", "candidate"):
            totals, kernels = [], []
            for runs in protocol["runs"]:
                prefix = root / "raw" / f"r{repeat}-{variant}-{case['name']}-n{runs}"
                log = text(Path(str(prefix) + ".log"))
                assert "Loaded 1 instances" in log
                assert "TENFERRO_CPU_BACKEND_KIND=blas" in log
                assert "TENFERRO_OPT_DOT_DECOMPOSER=0" in log
                assert "RAYON_NUM_THREADS=1, OMP_NUM_THREADS=1" in log
                assert f"of {runs} runs (1 warmup)" in log
                assert case["name"] in log and "panicked" not in log
                profile = text(Path(str(prefix) + ".callgrind"))
                assert re.search(r"^events: Ir$", profile, re.M)
                totals.append(int(re.search(r"^summary: (\d+)$", profile, re.M)[1]))
                annotation = text(Path(str(prefix) + "-annotated.txt"))
                kernels.append(sum(int(re.match(r"\s*([\d,]+)", line)[1].replace(",", "")) for line in annotation.splitlines() if "dgemm_kernel_HASWELL" in line))
            row[variant] = (totals[1] - totals[0]) / case["divisor"]
            row[variant + "_gemm_self"] = (kernels[1] - kernels[0]) / case["divisor"]
        row["reduction_percent"] = 100 * (1 - row["candidate"] / row["baseline"])
        samples.append(row)
    reductions = [s["reduction_percent"] for s in samples]
    results[case["name"]] = {
        "samples": samples,
        "baseline_median_Ir": statistics.median(s["baseline"] for s in samples),
        "candidate_median_Ir": statistics.median(s["candidate"] for s in samples),
        "median_paired_reduction_percent": statistics.median(reductions),
        "paired_reduction_range": [min(reductions), max(reductions)],
        "instruction_gate_passed": statistics.median(reductions) >= case["minimum_reduction_percent"],
    }
report = {"protocol": protocol, "results": results, "native_speedup": None}
(root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(results, indent=2))
