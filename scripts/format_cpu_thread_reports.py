#!/usr/bin/env python3
"""Combine explicitly selected CPU run reports without mixing commits or threads."""
import argparse
from pathlib import Path
import re

import yaml

REPORTS = {
    "einsum": ("report.md", "CPU Einsum Benchmark Results"),
    "cpu_ops": ("cpu_ops_report.md", "CPU Benchmark Results"),
    "linalg_jvp_vjp": ("linalg_jvp_vjp_report.md", "CPU Linalg JVP/VJP Benchmark Results"),
}


def combine(run_dirs, root=Path(".")):
    selected = []
    identities = set()
    thread_counts = set()
    root = root.resolve()
    for directory in run_dirs:
        directory = Path(directory).resolve()
        metadata = yaml.safe_load((directory / "run.yaml").read_text())
        thread = int(metadata["environment"]["env"]["RAYON_NUM_THREADS"])
        if thread in thread_counts:
            raise ValueError(f"duplicate thread count: {thread}")
        thread_counts.add(thread)
        identity = (metadata["target_profile"], metadata["tenferro_rs"]["commit"],
                    metadata["environment"]["env"].get("BENCHMARK_COMMIT"))
        identities.add(identity)
        selected.append((thread, directory))
    if not selected or len(identities) != 1:
        raise ValueError("select runs from exactly one target profile, tenferro commit, and benchmark commit")
    target, commit, benchmark_commit = identities.pop()
    # Every run is under this checkout, so source links remain reproducible.
    for _, directory in selected:
        directory.relative_to(root)
    destination = root / "result" / target / "cpu"
    destination.mkdir(parents=True, exist_ok=True)
    for name, (filename, title) in REPORTS.items():
        present = [(directory / filename).is_file() for _, directory in selected]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(f"missing {filename} in one selected run")
        lines = [f"# {title}", "", f"- Target profile: `{target}`",
                 f"- tenferro-rs commit: `{commit}`",
                 f"- Benchmark commit: `{benchmark_commit or 'unrecorded'}`", "",
                 "Generated from the explicit runs below; each section retains its own provenance and thread settings.", ""]
        for threads, directory in sorted(selected):
            source = directory / filename
            lines += [f"## Threads: {threads}", "", f"Source report: `{source.relative_to(root)}`.", ""]
            body = source.read_text().splitlines()[1:]
            for line in body:
                if line == f"## Threads: {threads}":
                    continue
                lines.append(re.sub(r"^(#{1,5}) ", r"#\1 ", line))
            lines.append("")
        (destination / f"{name}.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    combine([line for line in args.runs_manifest.read_text().splitlines() if line], args.root)
