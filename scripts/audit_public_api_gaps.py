#!/usr/bin/env python3
"""Find public tenferro session operations with no benchmark evidence.

The inventory is deliberately source-based: it reads the current
``extern/tenferro-rs`` checkout, then looks for each public session method in
the benchmark manifests, runners, or Rust benchmark binaries.  This catches a
new method even when nobody remembered to update the coverage YAML.

This is an inventory guard, not a claim that every spelling has a fair
cross-framework comparison.  The benchmark source still has to provide
equivalent PyTorch/JAX/Julia arms and matching setup/timing boundaries.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TENFERRO = ROOT / "extern/tenferro-rs"
SURFACES = (
    TENFERRO / "crates/tenferro-runtime/src/session_ext.rs",
    TENFERRO / "crates/tenferro-runtime/src/typed_session_ext.rs",
)
EVIDENCE_ROOTS = (ROOT / "benchmarks", ROOT / "scripts", ROOT / "src")


@dataclass(frozen=True)
class Api:
    name: str
    surfaces: tuple[str, ...]


def public_methods() -> list[Api]:
    found: dict[str, set[str]] = {}
    pattern = re.compile(r"^\s{4}fn\s+([A-Za-z_][A-Za-z0-9_]*)(?:<[^\n]+?>)?\s*\(", re.MULTILINE)
    for path in SURFACES:
        text = path.read_text()
        surface = path.name.removesuffix(".rs")
        for name in pattern.findall(text):
            found.setdefault(name, set()).add(surface)
    return [Api(name, tuple(sorted(surfaces))) for name, surfaces in sorted(found.items())]


def evidence(api: Api) -> list[str]:
    needle = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(api.name)}(?![A-Za-z0-9_])")
    hits: list[str] = []
    for root in EVIDENCE_ROOTS:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".rs", ".py", ".jl", ".yaml", ".yml", ".sh"}:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(errors="replace")
            if needle.search(text):
                hits.append(str(path.relative_to(ROOT)))
    return sorted(set(hits))


def rows() -> list[tuple[Api, list[str]]]:
    return [(api, evidence(api)) for api in public_methods()]


def render_markdown(items: list[tuple[Api, list[str]]]) -> str:
    lines = [
        "# Public session API benchmark gap audit",
        "",
        "Generated from the current `extern/tenferro-rs` checkout.",
        "",
        "| API | Surface(s) | Benchmark evidence | Status |",
        "|---|---|---|---|",
    ]
    for api, hits in items:
        evidence_text = ", ".join(f"`{hit}`" for hit in hits[:4]) or "—"
        status = "name evidence (execution unverified)" if hits else "NO NAME EVIDENCE"
        lines.append(f"| `{api.name}` | {', '.join(api.surfaces)} | {evidence_text} | {status} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail when any public method lacks evidence")
    parser.add_argument("--output", type=Path, help="write the Markdown inventory to this path")
    args = parser.parse_args()

    items = rows()
    report = render_markdown(items)
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")
    gaps = [(api, hits) for api, hits in items if not hits]
    if gaps:
        print(f"UNBENCHMARKED public session methods: {', '.join(api.name for api, _ in gaps)}", file=__import__("sys").stderr)
    return 1 if args.check and gaps else 0


if __name__ == "__main__":
    raise SystemExit(main())
