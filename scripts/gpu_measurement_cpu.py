#!/usr/bin/env python3
"""Choose the CPU a GPU benchmark arm runs on, from the CPUs this process may use.

A GPU arm's host thread issues the launches and waits on the device. When
``GPU_BENCH_CPU_SELECT=1``, the drivers (``scripts/run_gpu_suite.sh`` and
``scripts/run_gpu_public_api.sh``) call ``select`` before every arm. ``select``
scores every CPU in the allocation and picks the quietest. The arm is then
pinned to it with ``taskset``. Nothing about what is measured or verified
changes; only the CPU the arm runs on does (#335).

    python3 scripts/gpu_measurement_cpu.py select --record arms/pytorch-cuda.json
    python3 scripts/gpu_measurement_cpu.py record-run --run-metadata run.yaml arms/*.json

``select`` reads the allocation from the process affinity (``--cpuset`` or
``GPU_BENCH_CPUSET`` overrides it) and samples ``/proc/stat`` twice,
``--window`` seconds apart. It prints the chosen CPU and writes the per-CPU busy
fractions to ``--record``. If even the quietest CPU is busier than the threshold,
it exits with ``EXIT_NO_USABLE_CPU`` (76). Its message names the cpuset and every
CPU's busy fraction, so a driver can release the allocation and request another
one instead of sleeping. The threshold stays strict by default
(``DEFAULT_MAX_BUSY``). A run that relaxes it with ``GPU_BENCH_CPU_MAX_BUSY``
records that it did.

``record-run`` copies the per-arm records into ``run.yaml`` under
``collection.measurement_cpu``, next to the run's other collection facts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

EXIT_NO_USABLE_CPU = 76
DEFAULT_MAX_BUSY = 0.05
DEFAULT_WINDOW_SECONDS = 1.0


def parse_cpuset(text: str) -> list[int]:
    """``"0-3,8"`` -> ``[0, 1, 2, 3, 8]``; the kernel's list format."""
    cpus: set[int] = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        low, _, high = part.partition("-")
        start, end = int(low), int(high or low)
        if end < start:
            raise ValueError(f"invalid cpuset range: {part}")
        cpus.update(range(start, end + 1))
    if not cpus:
        raise ValueError(f"empty cpuset: {text!r}")
    return sorted(cpus)


def format_cpuset(cpus: list[int]) -> str:
    """The inverse of ``parse_cpuset``, with ranges collapsed."""
    parts, run = [], []
    for cpu in sorted(cpus):
        if run and cpu == run[-1] + 1:
            run.append(cpu)
            continue
        if run:
            parts.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [cpu]
    if run:
        parts.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(parts)


def read_stat(path: Path) -> dict[int, tuple[int, int]]:
    """``{cpu: (total, idle)}`` jiffies from a ``/proc/stat`` snapshot."""
    counters = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("cpu") or fields[0] == "cpu":
            continue
        values = [int(value) for value in fields[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
        counters[int(fields[0][3:])] = (sum(values), idle)
    return counters


def busy_fractions(first, second, cpus: list[int]) -> dict[int, float]:
    result = {}
    for cpu in cpus:
        if cpu not in first or cpu not in second:
            raise ValueError(f"cpu {cpu} of the allocation is not in /proc/stat")
        total = second[cpu][0] - first[cpu][0]
        idle = second[cpu][1] - first[cpu][1]
        result[cpu] = round((total - idle) / total, 4) if total > 0 else 0.0
    return result


def choose(busy: dict[int, float]) -> int:
    """The quietest CPU; a tie goes to the lowest id, so the choice is reproducible."""
    return min(busy, key=lambda cpu: (busy[cpu], cpu))


def allocation(cpuset: str | None) -> tuple[list[int], str]:
    if cpuset:
        return parse_cpuset(cpuset), "--cpuset"
    if os.environ.get("GPU_BENCH_CPUSET"):
        return parse_cpuset(os.environ["GPU_BENCH_CPUSET"]), "GPU_BENCH_CPUSET"
    return sorted(os.sched_getaffinity(0)), "process affinity"


def threshold(value: float | None) -> tuple[float, str]:
    if value is not None:
        return value, "--max-busy"
    if os.environ.get("GPU_BENCH_CPU_MAX_BUSY"):
        return float(os.environ["GPU_BENCH_CPU_MAX_BUSY"]), "GPU_BENCH_CPU_MAX_BUSY"
    return DEFAULT_MAX_BUSY, "default"


def select(args) -> int:
    cpus, source = allocation(args.cpuset)
    max_busy, max_busy_source = threshold(args.max_busy)
    window = args.window if args.window is not None else float(
        os.environ.get("GPU_BENCH_CPU_IDLE_WINDOW_SECONDS", DEFAULT_WINDOW_SECONDS))
    stat = Path(args.stat or os.environ.get("BENCHMARK_PROC_STAT", "/proc/stat"))
    first = read_stat(stat)
    time.sleep(window)
    second = read_stat(stat)
    busy = busy_fractions(first, second, cpus)
    cpu = choose(busy)
    usable = busy[cpu] <= max_busy
    record = {
        "arm": args.arm,
        "cpuset": format_cpuset(cpus),
        "cpuset_source": source,
        "window_seconds": window,
        "busy_fraction_by_cpu": {str(c): busy[c] for c in cpus},
        "cpu": cpu if usable else None,
        "busy_fraction": busy[cpu] if usable else None,
        "quietest_cpu": cpu,
        "max_busy": max_busy,
        "max_busy_source": max_busy_source,
        "status": "selected" if usable else "no_usable_cpu",
        "selected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        Path(args.record).write_text(json.dumps(record, indent=2) + "\n")
    if not usable:
        listing = ", ".join(f"cpu{c}={busy[c]:.3f}" for c in cpus)
        print(f"ERROR: no usable measurement CPU in cpuset {record['cpuset']} "
              f"({source}): the quietest, cpu{cpu}, was {busy[cpu]:.3f} busy over "
              f"{window:g} s (limit {max_busy:g}, {max_busy_source}). Per CPU: {listing}. "
              f"This allocation is the problem: release it and request another, or record a "
              f"relaxed GPU_BENCH_CPU_MAX_BUSY for the run.", file=sys.stderr)
        return EXIT_NO_USABLE_CPU
    print(cpu)
    return 0


def record_run(args) -> int:
    import yaml

    path = Path(args.run_metadata)
    metadata = yaml.safe_load(path.read_text()) or {}
    arms = {}
    for record_path in sorted(args.records):
        record = json.loads(Path(record_path).read_text())
        arms[record.get("arm") or Path(record_path).stem] = record
    collection = metadata.setdefault("collection", {})
    collection["measurement_cpu"] = {
        "policy": ("quietest CPU of the allocation per arm, pinned with taskset; "
                   f"exit {EXIT_NO_USABLE_CPU} when none is under the threshold")
        if any(r.get("status") in {"selected", "no_usable_cpu"} for r in arms.values())
        else "disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned",
        "arms": arms,
    }
    path.write_text(yaml.safe_dump(metadata, sort_keys=False))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("select", help="print the quietest usable CPU of the allocation")
    pick.add_argument("--cpuset")
    pick.add_argument("--max-busy", type=float)
    pick.add_argument("--window", type=float)
    pick.add_argument("--stat")
    pick.add_argument("--arm")
    pick.add_argument("--record")
    pick.set_defaults(run=select)
    merge = sub.add_parser("record-run", help="copy per-arm records into run.yaml")
    merge.add_argument("--run-metadata", required=True)
    merge.add_argument("records", nargs="*")
    merge.set_defaults(run=record_run)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
