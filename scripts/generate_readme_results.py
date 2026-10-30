#!/usr/bin/env python3
"""Generate the README "Latest Results" table from the published pages under ``result/``.

    uv run python scripts/generate_readme_results.py
    uv run python scripts/generate_readme_results.py --check

The table is the link index of the maintained reports: one row per
``result/<profile>/<device>/<suite>.md`` page of the ``maintained`` profiles in
``benchmarks/profiles.yaml`` plus the GPU profile.  It carries no measurements,
so the report file stays the only place a number is published.

Only the lines between ``<!-- results:begin -->`` and ``<!-- results:end -->``
are generated; the rest of the README is written by hand.  ``--check`` prints the
difference and exits 1 when the section is stale, and is the CI step that keeps
it from drifting.
"""

from __future__ import annotations

import argparse
import difflib
import pathlib
import sys
from typing import Iterable

BEGIN = "<!-- results:begin -->"
END = "<!-- results:end -->"
GPU_PROFILE = "nvidia-gpu"
PROFILES_FILE = pathlib.Path("benchmarks/profiles.yaml")
RESULT_DIR = pathlib.Path("result")
DEVICES = ("cpu", "gpu")
ROOT = pathlib.Path(__file__).resolve().parents[1]


class ReadmeError(RuntimeError):
    """The README or the profiles file cannot be generated from."""


def published_profiles(root: pathlib.Path = ROOT) -> list[str]:
    """The profiles whose pages the table lists: the maintained ones, then the GPU profile."""
    import yaml

    path = pathlib.Path(root) / PROFILES_FILE
    if not path.is_file():
        raise ReadmeError(f"{path}: no profile declaration")
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    maintained = [str(name) for name in document.get("maintained") or []]
    if not maintained:
        raise ReadmeError(f"{path}: declares no maintained profiles")
    return maintained + [profile for profile in (GPU_PROFILE,) if profile not in maintained]


def published_pages(root: pathlib.Path = ROOT) -> list[tuple[str, str, pathlib.Path]]:
    """[(profile, suite id, page path)] of the listed profiles, in profile and suite order."""
    root = pathlib.Path(root)
    pages: list[tuple[str, str, pathlib.Path]] = []
    for profile in published_profiles(root):
        for device in DEVICES:
            directory = root / RESULT_DIR / profile / device
            if not directory.is_dir():
                continue
            for page in sorted(directory.glob("*.md")):
                pages.append((profile, f"{device}/{page.stem}", page))
    return pages


def render_table(pages: Iterable[tuple[str, str, pathlib.Path]], root: pathlib.Path = ROOT) -> str:
    lines = ["| Target profile | Suite | Report |", "|---|---|---|"]
    rows = 0
    for profile, suite_id, page in pages:
        relative = page.resolve().relative_to(pathlib.Path(root).resolve()).as_posix()
        lines.append(f"| `{profile}` | `{suite_id}` | [{relative}]({relative}) |")
        rows += 1
    if not rows:
        lines.append("| - | - | No published pages under `result/` yet. |")
    return "\n".join(lines)


def generated_section(root: pathlib.Path = ROOT) -> str:
    return f"{BEGIN}\n{render_table(published_pages(root), root)}\n{END}"


def replace_section(text: str, section: str, *, source: str = "README.md") -> str:
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ReadmeError(f"{source}: expected one {BEGIN} and one {END}")
    head, _, rest = text.partition(BEGIN)
    _, _, tail = rest.partition(END)
    return f"{head}{section}{tail}"


def readme_path(root: pathlib.Path = ROOT) -> pathlib.Path:
    return pathlib.Path(root) / "README.md"


def update(root: pathlib.Path = ROOT) -> int:
    path = readme_path(root)
    current = path.read_text(encoding="utf-8")
    path.write_text(replace_section(current, generated_section(root), source=str(path)), encoding="utf-8")
    count = len(published_pages(root))
    print(f"wrote {path} ({count} pages)")
    return 0


def check(root: pathlib.Path = ROOT, stream=sys.stdout) -> int:
    path = readme_path(root)
    current = path.read_text(encoding="utf-8")
    expected = replace_section(current, generated_section(root), source=str(path))
    if current == expected:
        print(f"{path}: up to date ({len(published_pages(root))} pages)", file=stream)
        return 0
    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        expected.splitlines(keepends=True),
        fromfile=f"{path} (committed)",
        tofile=f"{path} (generated)",
    )
    stream.writelines(diff)
    print(f"{path}: results section is stale; run scripts/generate_readme_results.py", file=sys.stderr)
    return 1


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=pathlib.Path, default=ROOT,
                        help="repository root holding README.md, benchmarks/ and result/")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when the generated section differs from the committed one")
    args = parser.parse_args(argv)
    try:
        return check(args.root) if args.check else update(args.root)
    except ReadmeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
