"""Recover from stale ``torch.utils.cpp_extension`` JIT build locks.

``torch.utils.cpp_extension.load`` / ``load_inline`` guard a build directory
with a ``FileBaton`` lock at ``<build_directory>/lock``. ``FileBaton`` removes
the lock when the build finishes, but a build killed by a signal or a container
restart leaves the lock behind and the next invocation waits forever.

This module is intentionally free of third-party imports so it can be unit
tested (and used for diagnostics) without importing ``torch``. The build
directory resolution mirrors torch: ``TORCH_EXTENSIONS_DIR`` when set, otherwise
``torch.utils.cpp_extension._get_build_directory(name, verbose=False)``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

# A lock older than this with no live compiler is provably stale.
DEFAULT_STALE_AFTER_SECONDS = 120.0
# Upper bound for waiting on a lock held by a live compiler.
DEFAULT_WAIT_TIMEOUT_SECONDS = 1800.0
DEFAULT_POLL_SECONDS = 1.0

# Process names that a JIT extension build spawns. ``ninja`` drives the build;
# ``nvcc``/``cicc``/``cc1plus`` are the CUDA and host compiler stages.
_COMPILER_HINTS = ("nvcc", "ninja", "cc1plus", "cicc")


class ExtensionBuildLockError(RuntimeError):
    """Raised when a torch-extension build lock cannot be cleared in time."""


def resolve_build_directory(name: str) -> Path:
    """Resolve the build directory torch would use for extension ``name``."""
    try:
        from torch.utils import cpp_extension
    except Exception:
        cpp_extension = None

    if cpp_extension is not None and hasattr(cpp_extension, "_get_build_directory"):
        return Path(cpp_extension._get_build_directory(name, verbose=False))

    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not root:
        raise ExtensionBuildLockError(
            "cannot resolve torch extension build directory for "
            f"{name!r}: torch is not importable and TORCH_EXTENSIONS_DIR is unset")
    return Path(root) / name


def _iter_process_cmdlines() -> Iterable[tuple[int, str]]:
    """Yield ``(pid, cmdline)`` for live processes, stdlib only."""
    proc = Path("/proc")
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            if not raw:
                continue
            yield int(entry.name), raw.replace(b"\x00", b" ").decode("utf-8", "replace")
        return

    try:
        output = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, args = line.partition(" ")
        if pid_text.isdigit():
            yield int(pid_text), args


def live_compiler_pids(build_directory, *, cmdlines: Optional[Iterable] = None) -> list[int]:
    """PIDs of compiler processes whose command line mentions ``build_directory``."""
    build = os.path.realpath(str(build_directory))
    matches: list[int] = []
    source = cmdlines if cmdlines is not None else _iter_process_cmdlines()
    for pid, cmdline in source:
        if not any(hint in cmdline for hint in _COMPILER_HINTS):
            continue
        if build in cmdline:
            matches.append(int(pid))
    return matches


def ensure_build_lock_clear(
    name: str,
    *,
    stale_after: float = DEFAULT_STALE_AFTER_SECONDS,
    timeout: float = DEFAULT_WAIT_TIMEOUT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    build_directory=None,
    stream=sys.stderr,
    clock=time.monotonic,
    sleep=time.sleep,
    now=time.time,
    compiler_pids=None,
) -> Path:
    """Clear a provably stale torch-extension build lock before ``load``.

    Returns the resolved build directory. Raises ``ExtensionBuildLockError``
    when a live compiler holds the lock for longer than ``timeout`` instead of
    waiting forever in torch's ``FileBaton``.
    """
    if compiler_pids is None:
        compiler_pids = live_compiler_pids
    build = Path(build_directory) if build_directory is not None else resolve_build_directory(name)
    lock = build / "lock"
    deadline = clock() + timeout

    while True:
        if not lock.exists():
            return build
        alive = compiler_pids(build)
        try:
            age = max(0.0, now() - lock.stat().st_mtime)
        except FileNotFoundError:
            return build
        if not alive and age >= stale_after:
            print(
                f"[torch-extension-lock] removing stale build lock {lock} "
                f"(age {age:.0f}s, no live compiler)",
                file=stream,
                flush=True,
            )
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
            return build
        if clock() >= deadline:
            detail = ("a compiler is still running" if alive
                      else f"the lock is only {age:.0f}s old")
            raise ExtensionBuildLockError(
                f"timed out after {timeout:.0f}s waiting for torch extension "
                f"build lock {lock}: {detail}")
        sleep(poll_seconds)
