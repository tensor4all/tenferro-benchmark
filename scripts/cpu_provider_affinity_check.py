#!/usr/bin/env python3
"""Report per-thread CPU affinity groups while a command runs.

BLAS/LAPACK providers create their own thread teams, and those threads inherit
the CPU mask of the creating thread. A thread count therefore does not prove
that a provider runs on the CPUs the run intended: a worker confined to one CPU
confines the whole provider team. This helper records the observed
`Cpus_allowed_list` groups per thread comm so a run can show where provider and
worker threads were actually allowed to execute.

Usage:
    python3 scripts/cpu_provider_affinity_check.py --output FILE.json -- COMMAND...

The command is started with an inherited environment and the JSON summary is
written to --output (stdout is also printed).
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import time


def thread_masks(pid):
    """Return {comm: Counter(mask)} for every readable thread of ``pid``."""
    found = collections.defaultdict(collections.Counter)
    task_dir = f"/proc/{pid}/task"
    try:
        tids = os.listdir(task_dir)
    except OSError:
        return found
    for tid in tids:
        try:
            with open(f"{task_dir}/{tid}/status") as handle:
                comm = mask = None
                for line in handle:
                    if line.startswith("Name:"):
                        comm = line.split(":", 1)[1].strip()
                    elif line.startswith("Cpus_allowed_list:"):
                        mask = line.split(":", 1)[1].strip()
                    if comm is not None and mask is not None:
                        break
        except OSError:
            continue
        if comm is not None and mask is not None:
            found[comm][mask] += 1
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=0.005)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    masks = collections.defaultdict(collections.Counter)
    samples = 0
    while process.poll() is None:
        for comm, counter in thread_masks(process.pid).items():
            masks[comm].update(counter)
        samples += 1
        time.sleep(args.interval)
    try:
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        process.kill()
        stdout, stderr = process.communicate()
    for line in stdout.splitlines():
        print(line)

    summary = {
        "command": command,
        "exit_status": process.returncode,
        "signal": -process.returncode if process.returncode < 0 else None,
        "samples": samples,
        "threads": {
            comm: {
                "distinct_masks": {mask: count for mask, count in counter.most_common()},
                "allowed_cpu_counts": sorted(
                    {
                        (len(mask.split(",")) if mask != "" else 0)
                        for mask in counter
                    }
                ),
            }
            for comm, counter in sorted(masks.items())
        },
    }
    with open(args.output, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({k: summary[k] for k in ("exit_status", "signal", "samples")}))
    print(json.dumps(summary["threads"], sort_keys=True))
    if process.returncode < 0:  # propagate a signalled run
        sys.exit(128 - process.returncode)
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
