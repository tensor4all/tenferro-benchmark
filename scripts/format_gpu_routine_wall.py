"""Report complete GPU workflow cost and a descriptive hardware-specific budget."""
import argparse
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def render(selection_dir, budget):
    plan = json.loads((selection_dir / "selection.json").read_text())
    wall = json.loads((selection_dir / "wall_command_complete.json").read_text())
    core = json.loads((selection_dir / "case_status.json").read_text())
    seconds = wall["script_seconds_before_final_wall_record_write"]
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(budget["script_seconds"]) or budget["script_seconds"] <= 0:
        raise ValueError("invalid workflow wall time/budget")
    gpu_root = selection_dir.parent.parent
    suite_names = sorted({key.split(":")[0].split("/")[1] for key in plan["selected"]})
    metadata = yaml.safe_load((gpu_root / suite_names[0] / selection_dir.name / "run.yaml").read_text())
    checkpoints = [json.loads((gpu_root / name / selection_dir.name / "wall_command.json").read_text())
                   for name in suite_names]
    core_seconds = max(value["command_seconds_through_this_suite"] for value in checkpoints)
    controls = None
    if wall["routine_controls_selected"]:
        controls = json.loads((gpu_root / "routine_controls" / selection_dir.name / "case_status.json").read_text())
    applicable = (plan["coverage"] == budget["coverage"] and plan["effort"] == budget["effort"]
                  and metadata["target_profile"] == budget["target_profile"]
                  and metadata["cuda"]["device_name"] == budget["device_name"]
                  and core["complete"] and controls is not None and controls["complete"])
    lines = ["# GPU routine command wall", "", f"Raw selection: `{selection_dir.as_posix()}`.", "",
             f"tenferro-rs `{metadata['tenferro_rs']['commit']}`; harness `{metadata['harness']['commit']}`. "
             f"GPU: {metadata['cuda']['device_name']} (`{metadata['cuda']['device_uuid']}`).", "",
             f"Coverage `{plan['coverage']}`, effort `{plan['effort']}`; core measured "
             f"{len(core['executed'])}/{len(core['expected'])}. Routine controls selected: {wall['routine_controls_selected']}.", ""]
    if controls is not None:
        lines += [f"Controls measured {len(controls['measured'])}/{len(controls['timing'])}; "
                  f"state observations {len(controls['observed'])}/{len(controls['state'])}. "
                  "Untimed states are not operation timings.", ""]
    lines += ["| Command checkpoint/phase | Seconds |", "|---|---:|",
              f"| Core through final suite report | {core_seconds:.3f} |",
              f"| Remaining controls/accounting/reporting | {seconds-core_seconds:.3f} |",
              f"| Script through final wall checkpoint | {seconds:.3f} |", "",
              f"Scope: {wall['scope']}.", ""]
    outer = selection_dir / "wall_devcontainer_command.json"
    if outer.exists():
        measurement = json.loads(outer.read_text())
        lines += [f"External devcontainer command including CLI startup and final writes: "
                  f"**{measurement['seconds']:.2f} seconds**; tool `{measurement['tool']}`. "
                  "This is an independently measured enclosing interval, not an additive phase.", ""]
    lines += [f"Reference budget: **{budget['script_seconds']:g} seconds** for {budget['device_name']}, "
              f"{budget['coverage']}/{budget['effort']}. Conditions: {budget['cache_conditions']}.", "",
              f"Budget scope: {budget['scope']}. Reference characterization: `{budget['reference_run']}`.", ""]
    if applicable:
        lines += [f"Observation: **{'within budget' if seconds <= budget['script_seconds'] else 'over budget'}**.", ""]
    else:
        lines += ["Observation: budget not evaluated for this hardware, selection or incomplete/unmeasured run.", ""]
    lines += ["Budget status is descriptive and never changes functional success. A cold dependency build "
              "or busy host can exceed it; retain the observation rather than treating it as a confirmed regression. "
              "Detailed core case phases are in `routine_command_wall.md`; complete sequences and public state "
              "observations are in `routine_controls.md`. These observations establish no implementation speedup.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection_dir", type=Path)
    parser.add_argument("--budget", type=Path, default=ROOT / "benchmarks/gpu/routine_budget.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(args.selection_dir, yaml.safe_load(args.budget.read_text())))


if __name__ == "__main__":
    main()
