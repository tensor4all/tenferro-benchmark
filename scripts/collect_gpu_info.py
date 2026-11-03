#!/usr/bin/env python3
"""Collect CUDA GPU information for benchmark run metadata and reports."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
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


def detect_container(lock: Path = Path("/.dockerenv")) -> str:
    """Whether collection runs in a container: ``devcontainer`` or ``none``.

    The devcontainer is the recommended environment, not a requirement, so
    ``/.dockerenv`` is only detection and never gates collection.
    """
    return "devcontainer" if lock.exists() else "none"


def _cuda_toolkit_root() -> Path | None:
    """Toolkit root in the collectors' order: CUDA_HOME, CUDA_PATH, nvcc on PATH."""
    for name in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(name)
        if value and Path(value).is_dir():
            return Path(value)
    nvcc = shutil.which("nvcc")
    if nvcc:
        # Resolve an alternatives/module symlink to the toolkit it names.
        return Path(os.path.realpath(nvcc)).parent.parent
    return None


def _nvcc_version(root: Path | None) -> str | None:
    """Version of the selected toolkit's own nvcc, never an unrelated PATH one."""
    nvcc = root / "bin" / "nvcc" if root is not None else None
    if nvcc is None or not nvcc.is_file():
        return None
    output = _run([str(nvcc), "--version"])
    if not output:
        return None
    match = re.search(r"release ([0-9.]+)", output)
    return match.group(1) if match else None


def _cuda_runtime_version(root: Path | None = None) -> str | None:
    """CUDART version of the selected toolkit, else the loader's runtime.

    A compiler version does not identify a runtime tree, so nvcc is never a
    stand-in here: the toolkit version is recorded separately.
    """
    if root is None:
        root = _cuda_toolkit_root()
    candidates: list[str] = []
    if root is not None:
        candidates.extend(sorted(str(path) for path in (root / "lib64").glob("libcudart.so*")))
    candidates.extend(["libcudart.so.12", "libcudart.so.13"])
    for library in candidates:
        try:
            function = ctypes.CDLL(library).cudaRuntimeGetVersion
            function.argtypes = [ctypes.POINTER(ctypes.c_int)]
            function.restype = ctypes.c_int
            version = ctypes.c_int()
            if function(ctypes.byref(version)) == 0 and version.value > 0:
                return f"{version.value // 1000}.{(version.value % 1000) // 10}"
        except (OSError, AttributeError):
            continue
    return None


# Search order of the loaders at pins/tenferro-rs.rev: crates/tenferro-gpu
# `cuda/cutensor.rs` and crates/tenferro-linalg `gpu/ffi/cusolver.rs`. Each env
# override replaces the defaults, even when empty, and the first loadable
# candidate wins: an absent version symbol is unknown, not a reason to try the
# next candidate.
CUTENSOR_LIBRARY_PATHS = (
    "/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2",
    "libcutensor.so.2",
    "libcutensor.so",
)
CUSOLVER_LIBRARY_PATHS = (
    "libcusolver.so.12",
    "libcusolver.so.11",
    "/usr/lib/x86_64-linux-gnu/libcusolver.so.11",
    "libcusolver.so",
)


def _library_paths(env_var: str, defaults: tuple[str, ...]) -> list[str]:
    if env_var in os.environ:
        return [part for part in os.environ[env_var].split(":") if part]
    return list(defaults)


def _load_vendor_library(paths: list[str]) -> Any:
    for path in paths:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


def _cutensor_version() -> str | None:
    """cuTENSOR version through its public ``cutensorGetVersion()`` symbol."""
    library = _load_vendor_library(_library_paths("TENFERRO_CUTENSOR_PATH", CUTENSOR_LIBRARY_PATHS))
    if library is None:
        return None
    try:
        function = library.cutensorGetVersion
    except AttributeError:
        return None
    function.restype = ctypes.c_size_t
    function.argtypes = []
    try:
        value = int(function())
    except (OSError, ctypes.ArgumentError):
        return None
    if value <= 0:
        return None
    return f"{value // 10000}.{(value // 100) % 100}.{value % 100}"


def _cusolver_version() -> str | None:
    """cuSOLVER version through its public ``cusolverGetProperty`` API."""
    library = _load_vendor_library(_library_paths("TENFERRO_CUSOLVER_PATH", CUSOLVER_LIBRARY_PATHS))
    if library is None:
        return None
    try:
        function = library.cusolverGetProperty
    except AttributeError:
        return None
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    parts: list[int] = []
    for property_index in (0, 1, 2):  # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
        value = ctypes.c_int()
        try:
            status = function(property_index, ctypes.byref(value))
        except OSError:
            return None
        if status != 0:
            return None
        parts.append(value.value)
    return ".".join(str(part) for part in parts)


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

    toolkit_root = _cuda_toolkit_root()
    metadata: dict[str, Any] = {
        "device_ordinal": device["device_ordinal"],
        "device_name": device["device_name"],
        "device_uuid": device["device_uuid"],
        "device_memory_bytes": device["device_memory_bytes"],
        "driver_version": device["driver_version"],
        "runtime_version": _cuda_runtime_version(toolkit_root),
        "cuda_version": _nvidia_smi_cuda_version(),
        "toolkit_version": _nvcc_version(toolkit_root),
        "toolkit_root": str(toolkit_root) if toolkit_root is not None else None,
        "cutensor_version": _cutensor_version(),
        "cusolver_version": _cusolver_version(),
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
    for key, label in (
        ("toolkit_version", "CUDA toolkit"),
        ("toolkit_root", "CUDA toolkit root"),
        ("cutensor_version", "cuTENSOR version"),
        ("cusolver_version", "cuSOLVER version"),
    ):
        if metadata.get(key):
            info[label] = str(metadata[key])
    info["Container"] = detect_container()
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


def environment_info_from_run_metadata(run_metadata: dict[str, Any] | None) -> dict[str, str]:
    """Recorded collection environment, read only from the run record.

    Rendering a stored run must show the container and toolkit it was collected
    in, so this never probes the host. Keys an older record does not have are
    omitted rather than guessed.
    """
    run_metadata = run_metadata or {}
    info: dict[str, str] = {}
    environment = run_metadata.get("environment")
    if isinstance(environment, dict) and "container" in environment:
        info["Container"] = str(environment.get("container") or "unknown")
    cuda = run_metadata.get("cuda")
    if isinstance(cuda, dict):
        for key, label in (
            ("toolkit_version", "CUDA toolkit"),
            ("toolkit_root", "CUDA toolkit root"),
            ("cutensor_version", "cuTENSOR version"),
            ("cusolver_version", "cuSOLVER version"),
        ):
            if key in cuda:
                info[label] = str(cuda.get(key) or "unknown")
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
                info.update(environment_info_from_run_metadata(run_metadata))
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
