#!/usr/bin/env python3
"""Collect benchmark run metadata into a schema-validated YAML shape."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
from typing import Any

import yaml

from benchmark_layout import safe_suite_id_parts

ENV_KEYS = (
    "OMP_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "OPENBLAS_ROOT",
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
    parser.add_argument("--blas", choices=["openblas", "none"])
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
