"""Select and account for the routine public CUDA latency/state controls."""
import argparse
import json
import math
from pathlib import Path
import statistics

import yaml

ROOT = Path(__file__).resolve().parents[1]


def plan(coverage, effort):
    manifest = yaml.safe_load((ROOT / "benchmarks/gpu/manifests/routine_controls.yaml").read_text())
    cases = {row["id"]: row for row in json.loads((ROOT / manifest["source"]).read_text())}
    full = manifest["coverage"]["full"]
    for profile in manifest["coverage"].values():
        selected = profile["timing"] + profile["state"]
        if len(selected) != len(set(selected)) or not set(selected) <= cases.keys():
            raise ValueError("duplicate or unknown routine control")
        for lane in ("timing", "state"):
            if not set(profile[lane]) <= set(full[lane]):
                raise ValueError("quick control absent from full coverage")
            for key in profile[lane]:
                state = cases[key]["measurement_scope"] in {"state_diagnostic", "counter_diagnostic"}
                if state != (lane == "state"):
                    raise ValueError(f"control in wrong measurement lane: {key}")
    selected = manifest["coverage"][coverage]
    return dict(manifest_version=manifest["manifest_version"], coverage=coverage, effort=effort,
                **selected, unselected=[key for lane in full.values() for key in lane
                                       if key not in selected["timing"] + selected["state"]])


def account(selection, paths, verify_only=False):
    rows, duplicates, provenance = {}, [], []
    for path in paths:
        metadata = yaml.safe_load((path.parent / "run.yaml").read_text())
        provenance.append(dict(raw_run=path.parent.as_posix(), tenferro_rs=metadata["tenferro_rs"],
                               harness=metadata["harness"], cuda=metadata["cuda"]))
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = row["case_id"]
            if key in rows:
                duplicates.append(key)
            rows[key] = row
    expected = selection["timing"] + selection["state"]
    status = dict(selection, expected=expected, selected=expected,
                  measured=[], observed=[], verification_only=[], missing=[], failed=[], noisy=[],
                  duplicates=duplicates, unexpected=sorted(set(rows) - set(expected)), provenance=provenance)
    revisions = {(item["tenferro_rs"]["commit"], tuple(item["tenferro_rs"]["features"]),
                  item["harness"]["commit"]) for item in provenance}
    status["provenance_mismatch"] = len(revisions) != 1
    for key in expected:
        row = rows.get(key)
        if row is None:
            status["missing"].append(key)
            continue
        samples = row.get("samples", [])
        if row.get("correctness_status") != "passed":
            status["failed"].append(key)
        elif key in selection["state"]:
            status["failed" if samples else "observed"].append(key)
        elif verify_only:
            status["failed" if samples else "verification_only"].append(key)
        elif not samples or any(not math.isfinite(sample["elapsed_ns"]) or
                                sample["elapsed_ns"] <= 0 or not math.isfinite(sample["iterations"]) or
                                sample["iterations"] <= 0
                                for sample in samples):
            status["failed"].append(key)
        else:
            status["measured"].append(key)
            values = [sample["elapsed_ns"] / sample["iterations"] for sample in samples]
            if len(values) > 1 and statistics.stdev(values) / statistics.mean(values) > .10:
                status["noisy"].append(key)
    status["complete"] = (not verify_only and len(status["measured"]) == len(selection["timing"])
                          and len(status["observed"]) == len(selection["state"])
                          and not status["provenance_mismatch"]
                          and not any(status[key] for key in ("duplicates", "unexpected", "failed", "missing")))
    status["counts"] = {key: len(value) for key, value in status.items()
                        if isinstance(value, list) and key != "provenance"}
    return status


def report(status, source):
    lines = ["# GPU routine latency and state controls", "", f"Source: `{source}`.", "",
             f"Manifest {status['manifest_version']}; coverage {status['coverage']}; effort {status['effort']}.", "",
             "Timing controls measured and state controls observed: " + ("yes" if status["complete"] else "no") + ".", "",
             "State/counter controls contain no timed samples. Verification-only and unselected cases "
             "are not measured coverage. These controls do not establish a paired regression conclusion.", "",
             f"Provider/source mismatch across lanes: {status['provenance_mismatch']}.", ""]
    for source in status["provenance"]:
        lines += [f"Raw run: `{source['raw_run']}`; tenferro-rs `{source['tenferro_rs']['commit']}`; "
                  f"harness `{source['harness']['commit']}`; GPU UUID `{source['cuda']['device_uuid']}`.", ""]
    lines += ["| Classification | Count |", "|---|---:|"]
    lines += [f"| {key} | {count} |" for key, count in status["counts"].items()]
    for key in ("expected", "selected", "measured", "observed", "verification_only", "failed", "missing",
                "noisy", "unselected", "duplicates", "unexpected"):
        lines += ["", f"## {key}", ""] + ([f"- `{case}`" for case in status[key]] or ["None."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", choices=("quick", "full"), default="quick")
    parser.add_argument("--effort", default="standard")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", choices=("timing", "state"))
    parser.add_argument("--records", type=Path, nargs="+")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.records:
        selection = json.loads((args.output / "selection.json").read_text())
        status = account(selection, args.records, args.verify_only)
        (args.output / "case_status.json").write_text(json.dumps(status, indent=2) + "\n")
        (args.output / "report.md").write_text(report(status, args.output.as_posix()))
        if status["provenance_mismatch"] or any(status[key] for key in ("failed", "missing", "duplicates", "unexpected")):
            raise SystemExit("incomplete or failed routine GPU controls")
    else:
        selection = plan(args.coverage, args.effort)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        if args.lane:
            print(",".join(selection[args.lane]))


if __name__ == "__main__":
    main()
