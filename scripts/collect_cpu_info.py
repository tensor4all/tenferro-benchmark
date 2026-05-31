#!/usr/bin/env python3
"""Collect host CPU information for benchmark report metadata."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
from pathlib import Path


def _run(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(args, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    value = proc.stdout.strip()
    return value or None


def _cpuinfo_fields() -> dict[str, str]:
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return {}

    fields: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if key in {"model name", "cpu cores", "siblings", "vendor_id"} and key not in fields:
            fields[key] = value
    return fields


def _lscpu_fields() -> dict[str, str]:
    output = _run(["lscpu"])
    if not output:
        return {}

    wanted = {
        "Model name",
        "Vendor ID",
        "CPU(s)",
        "Core(s) per socket",
        "Socket(s)",
        "Thread(s) per core",
        "NUMA node(s)",
    }
    fields: dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if key in wanted:
            fields[key] = " ".join(value.split())
    return fields


def collect_cpu_info() -> dict[str, str]:
    lscpu = _lscpu_fields()
    cpuinfo = _cpuinfo_fields()

    info = {
        "Model": lscpu.get("Model name")
        or cpuinfo.get("model name")
        or platform.processor()
        or "unknown",
        "Vendor": lscpu.get("Vendor ID") or cpuinfo.get("vendor_id") or "unknown",
        "Logical CPUs": lscpu.get("CPU(s)") or str(os.cpu_count() or "unknown"),
        "Sockets": lscpu.get("Socket(s)") or "unknown",
        "Cores per socket": lscpu.get("Core(s) per socket") or cpuinfo.get("cpu cores") or "unknown",
        "Threads per core": lscpu.get("Thread(s) per core") or "unknown",
        "NUMA nodes": lscpu.get("NUMA node(s)") or "unknown",
        "Python platform": platform.platform(),
    }
    return info


def markdown(info: dict[str, str]) -> str:
    lines = ["## CPU Information", ""]
    for key, value in info.items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true", help="emit Markdown")
    args = parser.parse_args()

    info = collect_cpu_info()
    if args.markdown:
        print(markdown(info), end="")
    else:
        for key, value in info.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
