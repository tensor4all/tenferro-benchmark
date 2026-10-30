#!/usr/bin/env python3
"""Format observed command costs from explicitly selected CPU raw campaigns."""
import argparse
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = set()
    for directory in args.runs:
        for line in (directory / "cpu_info.md").read_text().splitlines():
            if line.startswith("- Model:"):
                models.add(line.split(":", 1)[1].strip())
    lines = ["# CPU public-route command wall observations", "",
             f"Sequential repository CPU devcontainer commands. Observed CPU models: {', '.join(sorted(models)) or 'unknown'}. These are command costs, not operation speedups. Times include build/checkout, metadata, prepared input generation, correctness, warmup, collection, separate diagnostics and report generation; final report file writes are excluded. Build cache state differs between campaigns. Initial reference runs preceded budget configuration. An over-budget command is diagnostic and never a functional failure.", "",
             "| Suite | Coverage | Effort | Features | Threads | Command s | Build s | Collection/metadata s | Diagnostic build s | Diagnostic collection s | Report s | Configured budget s | Budget status | Raw run |",
             "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for directory in args.runs:
        directory = directory.resolve()
        metadata_files = sorted(directory.glob("run_t*.yaml"))
        metadata = yaml.safe_load(metadata_files[0].read_text())
        collection = metadata["collection"]
        wall = json.loads((directory / "wall_command.json").read_text())
        phases = json.loads((directory / "wall_phases.json").read_text())
        threads = ",".join(p.stem.removeprefix("run_t") for p in metadata_files)
        features = ",".join(metadata["tenferro_rs"]["features"])
        command = wall.get("command_seconds_through_collection",
                           wall.get("command_seconds_before_final_report_write", 0.0))
        report = wall.get("report_generation_seconds")
        values = [metadata["suite_id"], collection["coverage"], collection["effort"], features, threads,
                  f"{command:.3f}",
                  f"{phases.get('build_and_checkout_seconds', phases.get('validation_and_latency_build_seconds', 0)):.3f}",
                  f"{phases.get('validation_metadata_and_collection_seconds', phases.get('metadata_and_latency_collection_seconds', 0)):.3f}",
                  f"{phases.get('allocation_diagnostic_build_seconds', 0):.3f}",
                  f"{phases.get('allocation_collection_seconds', 0):.3f}",
                  "—" if report is None else f"{report:.3f}", wall["budget_seconds"] or "unconfigured",
                  wall["budget_status"], f"`{directory.relative_to(ROOT)}`"]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
