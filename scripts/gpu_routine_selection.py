"""Versioned workload coverage and independent effort for routine CUDA suites."""
import argparse
import copy
import json
import math
import statistics
from pathlib import Path

import yaml

import bench_selection


def materialize(manifest_path, requested, output, coverage, backend_override="", problem="", effort="standard"):
    manifest = yaml.safe_load(manifest_path.read_text())
    root = manifest_path.resolve().parents[3]
    if coverage not in {"quick", "full"}:
        raise ValueError("GPU coverage must be quick or full")
    output.mkdir(parents=True, exist_ok=True)
    expected, selected, files, full = [], [], [], []
    requested = {(Path(path).resolve().relative_to(root).as_posix() if Path(path).is_absolute()
                  else Path(path).as_posix()) for path in requested}
    known_files = {spec["file"] for spec in manifest["suites"].values()}
    if requested - known_files:
        raise ValueError(f"suites absent from routine manifest: {sorted(requested-known_files)}")
    for suite_id, spec in manifest["suites"].items():
        if spec.get("optional") and spec["file"] not in requested:
            continue
        source = yaml.safe_load((root/spec["file"]).read_text())
        problems = {p["id"]:p for p in source["problems"]}
        if set(spec["full"]["problems"]) != problems.keys():
            raise ValueError(f"stale full problem manifest: {suite_id}")
        if not set(spec["quick"]["problems"]) <= problems.keys():
            raise ValueError(f"stale quick problem manifest: {suite_id}")
        for pid in spec["full"]["problems"]:
            full.extend(f"{suite_id}:{pid}:{backend}" for backend in spec["full"]["backends"])
        chosen = spec[coverage]
        backends = ([b for b in backend_override.split(",") if b in source["backends"]]
                    if backend_override else chosen["backends"])
        expected.extend(f"{suite_id}:{pid}:{backend}" for pid in chosen["problems"] for backend in backends)
        if spec["file"] not in requested or not chosen["problems"] or not backends:
            continue
        ids = [pid for pid in chosen["problems"] if not problem or pid == problem]
        if not ids:
            continue
        selected.extend(f"{suite_id}:{pid}:{backend}" for pid in ids for backend in backends)
        effective = copy.deepcopy(source)
        effective["backends"] = backends
        effective["problems"] = [copy.deepcopy(problems[pid]) for pid in ids]
        for item in effective["problems"]:
            run = dict(source.get("defaults", {}).get("run", {}), **item.get("run", {}))
            resolved = bench_selection.resolve_effort(run, value=effort)
            run.update(warmups=resolved["warmups"], runs=resolved["runs"])
            item["run"] = run
        path = output / (suite_id.replace("/", "_")+".yaml")
        path.write_text(yaml.safe_dump(effective, sort_keys=False, width=100))
        files.append(path.as_posix())
    if not selected:
        raise ValueError("no GPU cases selected; use full coverage for full-only workloads")
    plan = dict(manifest_version=manifest["manifest_version"], coverage=coverage, effort=effort,
                expected=expected, selected=selected, full=full,
                unselected=[key for key in full if key not in selected],
                suite_files=files, backend_override=backend_override or None,
                problem_filter=problem or None)
    (output/"selection.json").write_text(json.dumps(plan,indent=2)+"\n")
    return plan


def account(plan, paths):
    by_id, duplicates = {}, []
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            key = f"{row['suite_id']}:{row['problem_id']}:{row['backend']}"
            if key in by_id: duplicates.append(key)
            by_id[key] = row
    result = {name:[] for name in ("executed","unsupported","failed","missing","noisy","verification_only","raw_samples_unavailable")}
    for key in plan["selected"]:
        row = by_id.get(key)
        if row is None:
            result["missing"].append(key); continue
        if row["status"] in {"unsupported","not_configured"}:
            result["unsupported"].append(key); continue
        if row["status"] == "verification_only" and row["verification"]["status"] == "passed":
            result["verification_only"].append(key); continue
        if row["status"] != "ok" or row["verification"]["status"] != "passed":
            result["failed"].append(key); continue
        try:
            notes = json.loads(row["execution"].get("notes") or "{}")
            samples = notes["sampling"]["batch_ms"]
        except (KeyError,TypeError,ValueError):
            result["raw_samples_unavailable"].append(key); continue
        if not samples:
            result["raw_samples_unavailable"].append(key); continue
        if any(not math.isfinite(value) or value <= 0 for value in samples):
            result["failed"].append(key); continue
        result["executed"].append(key)
        if len(samples)>1 and statistics.stdev(samples)/statistics.mean(samples)>.10:
            result["noisy"].append(key)
    result.update(manifest_version=plan["manifest_version"],coverage=plan["coverage"],effort=plan["effort"],
                  expected=plan["expected"],selected=plan["selected"],unselected=plan["unselected"],
                  duplicates=duplicates,unexpected=sorted(set(by_id)-set(plan["selected"])))
    result["complete"] = (len(result["executed"])==len(plan["expected"]) and
                          not result["duplicates"] and not result["unexpected"])
    result["counts"] = {key:len(value) for key,value in result.items() if isinstance(value,list)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=Path("benchmarks/gpu/manifests/routine.yaml"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--coverage",default="quick")
    parser.add_argument("--backends",default="")
    parser.add_argument("--problem",default="")
    parser.add_argument("--effort",default="standard")
    parser.add_argument("--records",type=Path,nargs="*")
    parser.add_argument("suites",nargs="*")
    args = parser.parse_args()
    if args.records is not None:
        plan = json.loads((args.output/"selection.json").read_text())
        status = account(plan,args.records)
        (args.output/"case_status.json").write_text(json.dumps(status,indent=2)+"\n")
        if status["failed"] or status["missing"] or status["duplicates"] or status["unexpected"]:
            raise SystemExit("GPU collection has failed/missing/duplicate/unexpected cases")
        return
    plan = materialize(args.manifest,args.suites,args.output,args.coverage,args.backends,args.problem,args.effort)
    for path in plan["suite_files"]:
        print(path)


if __name__ == "__main__":
    main()
