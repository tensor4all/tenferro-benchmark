#!/usr/bin/env python3
"""Collect CUDA GPU information for benchmark run metadata and reports."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from typing import Any


def _run(args: list[str]) -> str | None:
    if shutil.which(args[0]) is None:
        return None
    try:
        proc = subprocess.run(args, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    value = proc.stdout.strip()
    return value or None


def _format_bytes(num_bytes: int) -> str:
    units = (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024))
    for unit, scale in units:
        if num_bytes % scale == 0 and num_bytes >= scale:
            return f"{num_bytes // scale} {unit}"
    return f"{num_bytes} B"


def _nvidia_smi_cuda_version() -> str | None:
    output = _run(["nvidia-smi"])
    if not output:
        return None
    match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
    return match.group(1) if match else None


def _nvcc_runtime_version() -> str | None:
    output = _run(["nvcc", "--version"])
    if not output:
        return None
    match = re.search(r"release ([0-9.]+)", output)
    return match.group(1) if match else None


def _torch_cudnn_version() -> str | None:
    probe = _run(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else '')",
        ]
    )
    if not probe:
        return None
    stripped = probe.strip()
    return stripped or None


def _query_nvidia_smi_device(device_ordinal: int) -> dict[str, Any] | None:
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
            "-i",
            str(device_ordinal),
        ]
    )
    if not output:
        return None

    row = next((line.strip() for line in output.splitlines() if line.strip()), "")
    if not row:
        return None

    parts = [part.strip() for part in row.split(",")]
    if len(parts) != 5:
        return None

    index_text, name, uuid, memory_mib, driver_version = parts
    try:
        index = int(index_text)
        memory_bytes = int(float(memory_mib)) * 1024 * 1024
    except ValueError:
        return None

    return {
        "device_ordinal": index,
        "device_name": name,
        "device_uuid": uuid,
        "device_memory_bytes": memory_bytes,
        "driver_version": driver_version,
    }


def collect_cuda_metadata(device_ordinal: int = 0) -> dict[str, Any] | None:
    device = _query_nvidia_smi_device(device_ordinal)
    if device is None:
        return None

    metadata: dict[str, Any] = {
        "device_ordinal": device["device_ordinal"],
        "device_name": device["device_name"],
        "device_uuid": device["device_uuid"],
        "device_memory_bytes": device["device_memory_bytes"],
        "driver_version": device["driver_version"],
        "runtime_version": _nvcc_runtime_version(),
        "cuda_version": _nvidia_smi_cuda_version(),
        "cudnn_version": _torch_cudnn_version(),
    }
    return metadata


def collect_gpu_info(device_ordinal: int = 0) -> dict[str, str]:
    metadata = collect_cuda_metadata(device_ordinal)
    if metadata is None:
        return {}

    info = {
        "Device": f"cuda:{metadata['device_ordinal']}",
        "Name": str(metadata["device_name"]),
        "UUID": str(metadata["device_uuid"]),
        "Memory": _format_bytes(int(metadata["device_memory_bytes"])),
        "Driver version": str(metadata["driver_version"] or "unknown"),
    }
    if metadata.get("cuda_version"):
        info["CUDA version"] = str(metadata["cuda_version"])
    if metadata.get("runtime_version"):
        info["CUDA runtime"] = str(metadata["runtime_version"])
    if metadata.get("cudnn_version"):
        info["cuDNN version"] = str(metadata["cudnn_version"])
    return info


def gpu_info_from_run_metadata(cuda: dict[str, Any]) -> dict[str, str]:
    if not cuda:
        return {}

    info: dict[str, str] = {}
    ordinal = cuda.get("device_ordinal")
    if ordinal is not None:
        info["Device"] = f"cuda:{ordinal}"
    if cuda.get("device_name"):
        info["Name"] = str(cuda["device_name"])
    if cuda.get("device_uuid"):
        info["UUID"] = str(cuda["device_uuid"])
    memory_bytes = cuda.get("device_memory_bytes")
    if isinstance(memory_bytes, int):
        info["Memory"] = _format_bytes(memory_bytes)
    if cuda.get("driver_version"):
        info["Driver version"] = str(cuda["driver_version"])
    if cuda.get("cuda_version"):
        info["CUDA version"] = str(cuda["cuda_version"])
    if cuda.get("runtime_version"):
        info["CUDA runtime"] = str(cuda["runtime_version"])
    if cuda.get("cudnn_version"):
        info["cuDNN version"] = str(cuda["cudnn_version"])
    return info


def gpu_info_from_records(records: list[dict[str, Any]]) -> dict[str, str]:
    for record in records:
        execution = record.get("execution") or {}
        environment = record.get("environment") or {}
        gpu_name = environment.get("gpu_name")
        if not gpu_name:
            continue

        info: dict[str, str] = {}
        ordinal = execution.get("device_ordinal")
        if ordinal is not None:
            info["Device"] = f"cuda:{ordinal}"
        info["Name"] = str(gpu_name)
        if environment.get("gpu_uuid"):
            info["UUID"] = str(environment["gpu_uuid"])
        memory_bytes = environment.get("gpu_memory_bytes")
        if isinstance(memory_bytes, int):
            info["Memory"] = _format_bytes(memory_bytes)
        if environment.get("cuda_version"):
            info["CUDA version"] = str(environment["cuda_version"])
        if environment.get("cudnn_version"):
            info["cuDNN version"] = str(environment["cudnn_version"])
        if environment.get("framework_version"):
            info["Framework"] = str(environment["framework_version"])
        return info
    return {}


def resolve_gpu_info(
    records: list[dict[str, Any]],
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    if run_metadata:
        cuda = run_metadata.get("cuda")
        if isinstance(cuda, dict):
            info = gpu_info_from_run_metadata(cuda)
            if info:
                return info

    info = gpu_info_from_records(records)
    if info:
        return info

    ordinal = 0
    for record in records:
        execution = record.get("execution") or {}
        if execution.get("device_ordinal") is not None:
            ordinal = int(execution["device_ordinal"])
            break
    return collect_gpu_info(ordinal)


def markdown(info: dict[str, str]) -> str:
    if not info:
        return ""
    lines = ["## GPU Information", ""]
    for key, value in info.items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument("--markdown", action="store_true", help="emit Markdown")
    args = parser.parse_args()

    info = collect_gpu_info(args.device_ordinal)
    if args.markdown:
        print(markdown(info), end="")
    else:
        for key, value in info.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
