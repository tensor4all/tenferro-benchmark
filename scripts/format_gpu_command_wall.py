#!/usr/bin/env python3
"""Descriptive GPU command/phase costs, independent of operation timing tables."""
import argparse
import json
import math
from pathlib import Path

import yaml


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def seconds(value):
    if value is None:
        return "—"
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid wall duration")
    return f"{value:.3f}"


def render(run_dirs, budget=None):
    if budget is not None and (not math.isfinite(budget) or budget <= 0):
        raise ValueError("budget must be finite and positive")
    lines = ["# GPU command-wall observations", "",
             "Wall time includes preparation, validation, sampling and reporting. "
             "These are end-to-end workflow costs; operation-performance ratios are not inferred. "
             "Missing phase instrumentation is displayed as unavailable, not zero. "
             "Filtered or unsuccessful runs do not establish full suite coverage.", ""]
    for path in run_dirs:
        metadata = yaml.safe_load((path / "run.yaml").read_text())
        command = json.loads((path / "wall_command.json").read_text())
        backends = rows(path / "wall_backends.jsonl")
        records = rows(path / "records.jsonl")
        keys = [(r["problem_id"], r["backend"]) for r in records]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate result rows")
        wall = command["command_seconds_through_this_suite"]
        lines += [f"## {metadata['suite_id']} / {path.name}", "",
                  f"Raw run: `{path.as_posix()}`.", "",
                  f"tenferro-rs `{metadata['tenferro_rs']['commit']}`; harness `{metadata['harness']['commit']}`. "
                  f"GPU: {metadata['cuda']['device_name']} (`{metadata['cuda']['device_uuid']}`).", "",
                  f"Command through this suite: **{seconds(wall)} s**; "
                  f"collection including build/release delays: {seconds(command['collection_seconds_including_build_and_backend_release_delays'])} s; "
                  f"report generation/copy: {seconds(command['report_generation_and_copy_seconds'])} s.", "",
                  f"Scope: {command['scope']}.", ""]
        if budget is not None:
            lines += [f"Declared descriptive budget: {budget:g} s; "
                      f"{'within budget' if wall <= budget else 'over budget'}. "
                      "Budget status does not change functional success or establish a regression.", ""]
        lines += ["| Backend | Build s | Process s (startup/setup/checks/samples) |",
                  "|---|---:|---:|"]
        for backend in backends:
            lines.append(f"| {backend['backend']} | {seconds(backend['build_seconds'])} | {seconds(backend['process_seconds'])} |")
        lines += ["", "Case phases below exclude process startup/JSONL writing. "
                  "Preparation includes input generation, transfers and reference preparation; "
                  "lazy initialization can occur in that phase. Sampler walls include untimed batch preparation/cleanup. "
                  "Command totals are cumulative across suites in one invocation, so do not sum those totals.", "",
                  "| Problem | Backend | Status | Backend init s | Prepare s | Compile s | Probe/init s | Warmup s | Calibration s | Sampling s | Validation s |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in records:
            try:
                notes = json.loads(row["execution"].get("notes") or "{}")
            except (ValueError, TypeError):
                notes = {}
            if not isinstance(notes, dict):
                notes = {}
            wall_phases = notes.get("wall_phases_ns", {})
            case = notes.get("case_wall_phases_ns", {})
            python = "case_body_before_json_write" in wall_phases
            if python:
                names = ["backend_initialization", "input_reference_and_preparation", "compilation",
                         "initialization_execution_and_placement", "sampler_warmup", "sampler_calibration",
                         "sampler_sampling", "initial_validation"]
                values = [wall_phases.get(key) for key in names]
                if values[-1] is not None:
                    values[-1] += wall_phases.get("post_sampling_execution_and_validation", 0)
            else:
                preparation = case.get("input_generation_transfer_reference_and_preparation",
                                       case.get("input_generation_transfer_and_preparation"))
                values = [case.get("backend_initialization"), preparation, None,
                          wall_phases.get("session_entry_and_initial_probe"), wall_phases.get("warmup"),
                          wall_phases.get("calibration"), wall_phases.get("sampling"), case.get("validation")]
            formatted = [seconds(None if value is None else value / 1e9) for value in values]
            lines.append("| " + " | ".join([row["problem_id"], row["backend"], row["status"], *formatted]) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--budget-seconds", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = render(args.run_dirs, args.budget_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)


if __name__ == "__main__":
    main()
