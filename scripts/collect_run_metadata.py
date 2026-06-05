#!/usr/bin/env python3
"""Collect benchmark run metadata into a schema-validated YAML shape."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
from typing import Any

import yaml

from benchmark_layout import safe_suite_id_parts

ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OMP_DYNAMIC",
    "RAYON_NUM_THREADS",
    "TENFERRO_CPU_BACKEND_KIND",
    "TENFERRO_OPT_DOT_DECOMPOSER",
    "OPENBLAS_ROOT",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "VECLIB_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "XLA_FLAGS",
    "CUDA_HOME",
    "USE_CUDA",
)


class MetadataError(Exception):
    """Raised when required metadata cannot be collected."""


def parse_features(values: list[str]) -> list[str]:
    features: list[str] = []
    for value in values:
        for feature in value.split(","):
            stripped = feature.strip()
            if stripped:
                features.append(stripped)
    return features


def run_git_rev_parse(tenferro_dir: Path) -> str:
    if not tenferro_dir.exists():
        raise MetadataError(f"tenferro dir does not exist: {tenferro_dir}")
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=tenferro_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise MetadataError("git is required to resolve tenferro commit") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        suffix = f": {detail}" if detail else ""
        raise MetadataError(
            f"failed to resolve tenferro commit in {tenferro_dir}{suffix}"
        ) from exc
    commit = result.stdout.strip()
    if not commit:
        raise MetadataError(f"git rev-parse returned an empty commit in {tenferro_dir}")
    return commit


def resolve_tenferro_commit(explicit_commit: str | None, tenferro_dir: Path | None) -> str:
    if explicit_commit:
        return explicit_commit
    if tenferro_dir is None:
        raise MetadataError(
            "--tenferro-commit is required when --tenferro-dir is not provided"
        )
    return run_git_rev_parse(tenferro_dir)


def non_empty_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def collect_environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "hostname": non_empty_or_none(socket.gethostname()),
        "os": non_empty_or_none(platform.platform()),
        "arch": non_empty_or_none(platform.machine()),
        "cpu": non_empty_or_none(platform.processor()),
    }
    env_values = {key: os.environ[key] for key in ENV_KEYS if key in os.environ}
    if env_values:
        environment["env"] = env_values
    return environment


def find_openblas_library(root: Path) -> str:
    candidates: list[Path] = []
    for lib_dir in (root / "lib", root / "lib64", root):
        if not lib_dir.exists():
            continue
        for pattern in (
            "libopenblas*.dylib",
            "libopenblas*.so",
            "libopenblas*.a",
            "openblas*.lib",
        ):
            candidates.extend(sorted(lib_dir.glob(pattern)))
    if candidates:
        return str(candidates[0])
    return str(root)


def parse_openblas_version_header(path: Path) -> str | None:
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return None
    for line in text.splitlines():
        if "OPENBLAS_VERSION" not in line:
            continue
        quoted = re.search(r'"([^"]+)"', line)
        if quoted:
            version = quoted.group(1).strip()
            if version:
                return version
        tokens = line.split()
        if tokens:
            version = tokens[-1].strip()
            if version:
                return version
    return None


def find_openblas_version(root: Path) -> str:
    for header in (
        root / "include" / "openblas_config.h",
        root / "include" / "openblas" / "openblas_config.h",
    ):
        version = parse_openblas_version_header(header)
        if version:
            return version
    return "unknown"


def collect_blas(blas: str | None) -> dict[str, str] | None:
    if blas is None:
        return None
    if blas == "none":
        return {"implementation": "none"}
    if blas == "accelerate":
        return {
            "implementation": "accelerate",
            "version": "unknown",
            "library": "/System/Library/Frameworks/Accelerate.framework",
        }

    root_value = os.environ.get("OPENBLAS_ROOT")
    if not root_value:
        raise MetadataError("--blas openblas requires OPENBLAS_ROOT to be set")
    root = Path(root_value).expanduser()
    return {
        "implementation": "openblas",
        "version": find_openblas_version(root),
        "root": str(root),
        "library": find_openblas_library(root),
    }


def run_python_probe(source: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", source],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "available": False,
            "reason": f"invalid probe JSON: {exc}",
        }
    return parsed if isinstance(parsed, dict) else {"available": False, "reason": "probe did not return an object"}


def linked_libraries(path: str) -> list[str]:
    library = Path(path)
    if not library.exists():
        return []
    if platform.system() == "Darwin" and shutil.which("otool"):
        args = ["otool", "-L", str(library)]
    elif shutil.which("ldd"):
        args = ["ldd", str(library)]
    else:
        return []
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    deps: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.endswith(":"):
            continue
        dep = stripped.split(" (", 1)[0].strip()
        if "=>" in dep:
            dep = dep.split("=>", 1)[1].strip().split(" ", 1)[0]
        if dep:
            deps.append(dep)
    return deps


def relevant_linked_libraries(paths: list[str]) -> list[str]:
    tokens = ("accelerate", "veclib", "openblas", "mkl", "blas", "lapack", "omp")
    selected: list[str] = []
    for path in paths:
        for dep in linked_libraries(path):
            if any(token in dep.lower() for token in tokens) and dep not in selected:
                selected.append(dep)
    return selected


def detect_provider(*parts: str | None) -> str:
    text = "\n".join(part for part in parts if part).lower()
    if "openblas" in text:
        return "openblas"
    if "accelerate.framework" in text or "veclib" in text or "blas_info=accelerate" in text:
        return "accelerate"
    if "mkl" in text:
        return "mkl"
    if "_lapack" in text:
        return "internal_lapack"
    if "blas" in text or "lapack" in text:
        return "blas_lapack"
    return "none_detected"


def jax_xla_backend_name(backend: Any) -> str:
    value = non_empty_or_none(str(backend)) if backend is not None else None
    if value is None:
        return "xla_unknown"
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return f"xla_{normalized}" if normalized else "xla_unknown"


def torch_config_value(config: str, key: str) -> str | None:
    match = re.search(rf"\b{re.escape(key)}=([^,\n]+)", config)
    if match:
        return match.group(1).strip()
    return None


def collect_pytorch_backend() -> dict[str, Any]:
    info = run_python_probe(
        r'''
import json
from pathlib import Path

try:
    import torch
    base = Path(torch.__file__).resolve().parent
    libraries = sorted(str(path.resolve()) for path in base.glob("lib/libtorch_cpu*") if path.is_file())
    print(json.dumps({
        "available": True,
        "version": str(torch.__version__),
        "file": str(Path(torch.__file__).resolve()),
        "config": torch.__config__.show(),
        "libraries": libraries,
    }))
except Exception as exc:
    print(json.dumps({
        "available": False,
        "reason": f"{type(exc).__name__}: {exc}",
    }))
'''
    )
    if not info.get("available"):
        return {
            "available": False,
            "reason": str(info.get("reason", "unavailable")),
        }

    config = str(info.get("config", ""))
    libraries = [str(path) for path in info.get("libraries", []) if isinstance(path, str)]
    deps = relevant_linked_libraries(libraries)
    blas_info = torch_config_value(config, "BLAS_INFO")
    lapack_info = torch_config_value(config, "LAPACK_INFO")
    provider = detect_provider(blas_info, lapack_info, "\n".join(deps), config)
    version = info.get("version")
    file_path = info.get("file")
    return {
        "available": True,
        "version": non_empty_or_none(str(version)) if version is not None else None,
        "file": non_empty_or_none(str(file_path)) if file_path is not None else None,
        "provider": provider,
        "blas_info": blas_info,
        "lapack_info": lapack_info,
        "library": libraries[0] if libraries else None,
        "linked_libraries": deps,
    }


def collect_jax_backend() -> dict[str, Any]:
    info = run_python_probe(
        r'''
import json
from pathlib import Path

try:
    import jax
    import jaxlib
    base = Path(jaxlib.__file__).resolve().parent
    libraries = []
    for pattern in ("*.so", "*.dylib", "**/*.so", "**/*.dylib"):
        libraries.extend(str(path.resolve()) for path in base.glob(pattern) if path.is_file())
    libraries = sorted(set(libraries))
    try:
        backend = jax.default_backend()
    except Exception:
        backend = None
    print(json.dumps({
        "available": True,
        "version": str(jax.__version__),
        "jaxlib_version": str(jaxlib.__version__),
        "file": str(Path(jaxlib.__file__).resolve()),
        "backend": backend,
        "libraries": libraries[:64],
    }))
except Exception as exc:
    print(json.dumps({
        "available": False,
        "reason": f"{type(exc).__name__}: {exc}",
    }))
'''
    )
    if not info.get("available"):
        return {
            "available": False,
            "reason": str(info.get("reason", "unavailable")),
        }

    libraries = [str(path) for path in info.get("libraries", []) if isinstance(path, str)]
    deps = relevant_linked_libraries(libraries)
    lapack_provider = detect_provider("\n".join(deps))
    version = info.get("version")
    jaxlib_version = info.get("jaxlib_version")
    backend = info.get("backend")
    file_path = info.get("file")
    dot_backend = jax_xla_backend_name(backend)
    return {
        "available": True,
        "version": non_empty_or_none(str(version)) if version is not None else None,
        "jaxlib_version": non_empty_or_none(str(jaxlib_version)) if jaxlib_version is not None else None,
        "backend": non_empty_or_none(str(backend)) if backend is not None else None,
        "file": non_empty_or_none(str(file_path)) if file_path is not None else None,
        "provider": dot_backend,
        "dot_backend": dot_backend,
        "lapack_provider": lapack_provider,
        "library": libraries[0] if libraries else None,
        "linked_libraries": deps,
    }


def collect_python_backends() -> dict[str, Any]:
    return {
        "pytorch": collect_pytorch_backend(),
        "jax": collect_jax_backend(),
    }


def build_metadata(args: argparse.Namespace) -> dict[str, Any]:
    safe_suite_id_parts(args.suite_id)
    tenferro_dir = args.tenferro_dir.expanduser() if args.tenferro_dir else None
    commit = resolve_tenferro_commit(args.tenferro_commit, tenferro_dir)
    tenferro_path = str(tenferro_dir) if tenferro_dir else "unknown"

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "suite_id": args.suite_id,
        "suite_file": str(args.suite_file),
        "timestamp": args.timestamp,
        "tenferro_rs": {
            "path": tenferro_path,
            "commit": commit,
            "features": parse_features(args.features),
        },
        "environment": collect_environment(),
    }
    blas = collect_blas(args.blas)
    if blas is not None:
        metadata["blas"] = blas
    metadata["python_backends"] = collect_python_backends()
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--suite-file", required=True, type=Path)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--tenferro-dir", type=Path)
    parser.add_argument("--tenferro-commit")
    parser.add_argument("--tenferro-ref")
    parser.add_argument("--features", action="append", default=[])
    parser.add_argument("--blas", choices=["openblas", "accelerate", "none"])
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def write_yaml(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(metadata, fh, sort_keys=False)


def main() -> int:
    args = parse_args()
    try:
        metadata = build_metadata(args)
        write_yaml(args.output, metadata)
    except (MetadataError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
