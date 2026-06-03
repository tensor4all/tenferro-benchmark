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


def _sysctl_value(key: str) -> str | None:
    return _run(["sysctl", "-n", key])


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _format_bytes(value: str | None) -> str | None:
    size = _int_or_none(value)
    if size is None:
        return None
    units = (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024))
    for unit, scale in units:
        if size % scale == 0 and size >= scale:
            return f"{size // scale} {unit}"
    return f"{size} B"


def _darwin_perflevels() -> str:
    count = _int_or_none(_sysctl_value("hw.nperflevels")) or 0
    levels: list[str] = []
    for index in range(count):
        prefix = f"hw.perflevel{index}"
        name = _sysctl_value(f"{prefix}.name") or f"perflevel{index}"
        physical = _sysctl_value(f"{prefix}.physicalcpu") or "unknown"
        logical = _sysctl_value(f"{prefix}.logicalcpu") or "unknown"
        parts = [f"{name}: {physical} physical / {logical} logical"]
        l1i = _format_bytes(_sysctl_value(f"{prefix}.l1icachesize"))
        l1d = _format_bytes(_sysctl_value(f"{prefix}.l1dcachesize"))
        l2 = _format_bytes(_sysctl_value(f"{prefix}.l2cachesize"))
        cpus_per_l2 = _sysctl_value(f"{prefix}.cpusperl2")
        cache_parts = []
        if l1i:
            cache_parts.append(f"L1i {l1i}")
        if l1d:
            cache_parts.append(f"L1d {l1d}")
        if l2:
            cache_parts.append(f"L2 {l2}")
        if cpus_per_l2:
            cache_parts.append(f"{cpus_per_l2} CPUs/L2")
        if cache_parts:
            parts.append(f"({', '.join(cache_parts)})")
        levels.append(" ".join(parts))
    return "; ".join(levels) if levels else "unknown"


def _darwin_cpu_info() -> dict[str, str]:
    model = _sysctl_value("machdep.cpu.brand_string") or platform.processor() or "unknown"
    physical = _sysctl_value("hw.physicalcpu") or "unknown"
    logical = _sysctl_value("hw.logicalcpu") or str(os.cpu_count() or "unknown")
    physical_int = _int_or_none(physical)
    logical_int = _int_or_none(logical)
    threads_per_core = "unknown"
    if physical_int and logical_int and logical_int % physical_int == 0:
        threads_per_core = str(logical_int // physical_int)

    return {
        "Model": model,
        "Vendor": "Apple" if model.startswith("Apple ") else "unknown",
        "Logical CPUs": logical,
        "Physical CPUs": physical,
        "Sockets": "1",
        "Cores per socket": physical,
        "Threads per core": threads_per_core,
        "NUMA nodes": "1",
        "Performance levels": _darwin_perflevels(),
        "Python platform": platform.platform(),
    }


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
    if platform.system() == "Darwin":
        return _darwin_cpu_info()

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
