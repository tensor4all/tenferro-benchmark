#!/usr/bin/env python3
"""Sequential same-build A/A and balanced Rust/PyTorch MWE confirmation.

This is a cross-implementation comparison, not a revision-regression detector.
The declaration follows the fields in benchmarks/cpu/confirmation.yaml and is
written after A/A and before collecting the PyTorch candidate measurements.
"""
import os
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone
import yaml

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = ("ifft", "diagonal", "triangular", "lstsq")


def metrics(row):
    values = [r["elapsed_ns"]/r["iterations"] for r in row["samples"]]
    return statistics.median(values), statistics.stdev(values)/statistics.mean(values)


def collect(folder, phase, op, threads, round_index, arm, implementation):
    filename = folder / f"{phase}_{op}_t{threads}_r{round_index}_{arm}.json"
    if implementation == "rust":
        command = [str(ROOT/"target/compatibility-mkl/release/examples/cpu_gap_mwe"), op, str(threads)]
    else:
        command = [str(ROOT/".venv/bin/python"), "scripts/cpu_gap_mwe_python.py", op, str(threads)]
    prefix = f'''set -euo pipefail
export LD_LIBRARY_PATH="$MKLROOT/lib:/opt/intel/oneapi/compiler/2026.1/lib:${{LD_LIBRARY_PATH:-}}"
source scripts/thread_env.sh
configure_cpu_thread_env {threads}
source scripts/benchmark_host_idle.sh
assert_benchmark_host_idle
exec "$@"
'''
    completed = subprocess.run(["bash", "-lc", prefix, "--", *command], cwd=ROOT,
                               text=True, capture_output=True, check=True)
    row = json.loads(completed.stdout)
    assert row["correctness_status"] == "passed"
    row.update(phase=phase, operation=op, threads=threads, round=round_index,
               arm=arm, implementation=implementation, command=command)
    filename.write_text(json.dumps(row, indent=2)+"\n")
    return row


def main():
    folder = Path(sys.argv[1]).resolve()
    operations = tuple(sys.argv[2:]) or OPERATIONS
    folder.mkdir(parents=True, exist_ok=True)
    declaration = folder / "confirmation.yaml"
    if declaration.exists():
        raise RuntimeError("Refusing to overwrite an existing confirmation declaration")
    aa = {}
    for op in operations:
        for threads in (1, 4):
            pairs = []
            for round_index in range(4):
                order = ("A", "B") if round_index % 2 == 0 else ("B", "A")
                rows = {arm: collect(folder, "aa", op, threads, round_index, arm, "rust")
                        for arm in order}
                pairs.append(metrics(rows["A"])[0]/metrics(rows["B"])[0])
            aa[f"{op}@t{threads}"] = pairs
            print("A/A", op, threads, pairs, flush=True)
    library_commit = subprocess.check_output(["git", "-C", "extern/tenferro-rs", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    harness_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = yaml.safe_load((ROOT/"benchmarks/cpu/confirmation.yaml").read_text())
    config.update(status="declared", declared_before_candidate_results=datetime.now(timezone.utc).isoformat(),
        library={"baseline_commit": library_commit, "candidate_commit": library_commit},
        harness_commit=harness_commit, build={"profile": "release", "features": ["blas-mkl"]},
        host={"target_profile": "amd-cpu", "hostname": subprocess.check_output(["hostname"], text=True).strip(),
              "affinity": "none (thread env only)", "provider": "MKL; versions independently recorded"},
        threads=[1, 4], cases={"suite_id": "cpu/perf_issues", "manifest_version": "new MWE cases pending issue assignment",
                              "coverage": "explicit", "case_ids": list(operations)},
        timing={"scope": "many_operations_single_interval; session/setup/cleanup outside", "cache_pool_state": "warm, validated and primed", "target_interval_ns": int(os.environ.get("CPU_GAP_TARGET_NS", "2000000"))},
        repetitions={"warmups": 3, "runs": 15, "rounds": 4}, statistic="median_of_round_ratios",
        thresholds={"relative": 0.20, "absolute_ns": 500},
        noise={"max_cov": 0.20, "max_aa_relative_spread": 0.10, "host_idle_guard": "enabled, never bypassed"},
        comparison={"baseline": "PyTorch Python", "candidate": "tenferro-rs borrowed-session public API",
                    "kind": "cross-implementation; not historical revision regression",
                    "threshold_basis": "20% practical gap and 500ns minimum; A/A gate independently checks stability",
                    "aa_ratios": aa})
    declaration.write_text(yaml.safe_dump(config, sort_keys=False))
    declaration_hash = hashlib.sha256(declaration.read_bytes()).hexdigest()
    results = []
    for op in operations:
        for threads in (1, 4):
            pairs, rust_ns, torch_ns, covs = [], [], [], []
            for round_index in range(4):
                order = ("A", "B") if round_index % 2 == 0 else ("B", "A")
                rows = {arm: collect(folder, "paired", op, threads, round_index, arm,
                                    "rust" if arm == "A" else "pytorch") for arm in order}
                a, ac = metrics(rows["A"]); b, bc = metrics(rows["B"])
                pairs.append(a/b); rust_ns.append(a); torch_ns.append(b); covs.extend((ac,bc))
            ratio = statistics.median(pairs)
            aa_statistic = statistics.median(aa[f"{op}@t{threads}"])
            stable = abs(aa_statistic-1) <= 0.10 and max(covs) <= 0.20
            slower = ratio > 1.20 and statistics.median(rust_ns)-statistics.median(torch_ns)>500
            row = dict(operation=op, threads=threads, rust_ns=statistics.median(rust_ns),
                       pytorch_ns=statistics.median(torch_ns), ratios=pairs, ratio=ratio,
                       aa_statistic=aa_statistic, max_cov=max(covs),
                       verdict="INCONCLUSIVE" if not stable else "CONFIRMED_SLOWER" if slower else "NO_MATERIAL_GAP")
            results.append(row); print(row, flush=True)
    assert hashlib.sha256(declaration.read_bytes()).hexdigest() == declaration_hash
    (folder/"confirmation.json").write_text(json.dumps({"config_sha256": declaration_hash, "results": results}, indent=2)+"\n")


if __name__ == "__main__":
    main()
