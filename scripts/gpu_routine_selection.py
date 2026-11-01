"""Materialize and account for the single routine CUDA suite set."""
import argparse
import copy
import json
import math
import statistics
from pathlib import Path

import yaml

import bench_selection

DEFAULT_DRIVER = "gpu_suite"


def defaults(manifest_path, vendor_arms=False):
    """The manifest's own suite files and the backend arm set it declares."""
    manifest = yaml.safe_load(manifest_path.read_text())
    suites = [spec["file"] for spec in manifest["suites"].values()]
    backends = []
    for spec in manifest["suites"].values():
        declared = list(spec["backends"])
        if vendor_arms:
            declared += list(spec.get("vendor_backends") or [])
        backends.extend(backend for backend in declared if backend not in backends)
    return suites, backends


def materialize(manifest_path, requested, output, backend_override="", problem="", effort="standard",
                vendor_arms=False):
    manifest = yaml.safe_load(manifest_path.read_text())
    root = manifest_path.resolve().parents[3]
    output.mkdir(parents=True, exist_ok=True)
    expected, selected, files, drivers = [], [], [], {}
    requested = {(Path(path).resolve().relative_to(root).as_posix() if Path(path).is_absolute()
                  else Path(path).as_posix()) for path in requested}
    known_files = {spec["file"] for spec in manifest["suites"].values()}
    if requested - known_files:
        raise ValueError(f"suites absent from routine manifest: {sorted(requested-known_files)}")
    for suite_id, spec in manifest["suites"].items():
        source = yaml.safe_load((root/spec["file"]).read_text())
        problems = {p["id"]:p for p in source["problems"]}
        if not set(spec["problems"]) <= problems.keys():
            raise ValueError(f"stale routine problem manifest: {suite_id}")
        backends = list(spec["backends"])
        if vendor_arms:
            backends += list(spec.get("vendor_backends") or [])
        if not set(backends) <= set(source["backends"]):
            raise ValueError(f"routine backends absent from {spec['file']}: {suite_id}")
        if backend_override:
            backends = [b for b in backend_override.split(",") if b in source["backends"]]
        expected.extend(f"{suite_id}:{pid}:{backend}" for pid in spec["problems"] for backend in backends)
        if spec["file"] not in requested or not backends:
            continue
        ids = [pid for pid in spec["problems"] if not problem or pid == problem]
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
        drivers[path.as_posix()] = spec.get("driver") or DEFAULT_DRIVER
    if not selected:
        raise ValueError("no GPU cases selected")
    plan = dict(manifest_version=manifest["manifest_version"], effort=effort,
                expected=expected, selected=selected, suite_files=files, suite_drivers=drivers,
                backend_override=backend_override or None, vendor_arms=vendor_arms,
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
    result = {name:[] for name in ("executed","skipped","unsupported","failed","missing","noisy","verification_only","raw_samples_unavailable")}
    reasons = {}
    for key in plan["selected"]:
        row = by_id.get(key)
        if row is None:
            result["missing"].append(key); continue
        verification = row.get("verification", {})
        reason = next((value.strip() for value in (
            row.get("execution", {}).get("unsupported_reason"), verification.get("reason"))
            if isinstance(value, str) and value.strip()), None)
        if verification.get("status") == "failed":
            result["failed"].append(key)
            reasons[key] = reason or "numerical verification failed"
            continue
        if row["status"] in {"skipped", "unsupported", "not_configured"}:
            if reason is None:
                result["failed"].append(key)
                reasons[key] = f"{row['status']} requires a nonblank reason"
            else:
                category = "unsupported" if row["status"] == "unsupported" else "skipped"
                result[category].append(key)
                reasons[key] = reason
            continue
        if row["status"] == "verification_only" and row["verification"]["status"] == "passed":
            result["verification_only"].append(key); continue
        if row["status"] != "ok" or verification.get("status") != "passed":
            result["failed"].append(key)
            reasons[key] = reason or f"status={row['status']}, verification={verification.get('status')}"
            continue
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
    result.update(manifest_version=plan["manifest_version"],effort=plan["effort"],
                  expected=plan["expected"],selected=plan["selected"],
                  duplicates=duplicates,unexpected=sorted(set(by_id)-set(plan["selected"])),
                  reasons=reasons)
    result["complete"] = (len(result["executed"])==len(plan["expected"]) and
                          not result["duplicates"] and not result["unexpected"])
    result["counts"] = {key:len(value) for key,value in result.items() if isinstance(value,list)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=Path("benchmarks/gpu/manifests/routine.yaml"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--backends",default="")
    parser.add_argument("--problem",default="")
    parser.add_argument("--effort",default="standard")
    parser.add_argument("--vendor-arms",action="store_true",
        help="add the opt-in vendor arms declared next to each routine suite")
    parser.add_argument("--records",type=Path,nargs="*")
    parser.add_argument("--defaults",action="store_true",
        help="print the manifest's suite files and backend arms instead of selecting")
    parser.add_argument("suites",nargs="*")
    args = parser.parse_args()
    if args.defaults:
        suites, backends = defaults(args.manifest, args.vendor_arms)
        print(",".join(suites))
        print(",".join(backends))
        return
    if args.output is None:
        parser.error("--output is required when --defaults is not used")
    if args.records is not None:
        plan = json.loads((args.output/"selection.json").read_text())
        status = account(plan,args.records)
        (args.output/"case_status.json").write_text(json.dumps(status,indent=2)+"\n")
        for category in ("failed", "skipped", "unsupported", "missing", "duplicates", "unexpected"):
            print(f"{category}: {status['counts'][category]}")
            for key in status[category]:
                print(f"  {key}: {status['reasons'].get(key, category)}")
        if status["failed"] or status["missing"] or status["duplicates"] or status["unexpected"]:
            raise SystemExit("GPU collection has failed/missing/duplicate/unexpected cases")
        return
    plan = materialize(args.manifest,args.suites,args.output,args.backends,args.problem,args.effort,
                       args.vendor_arms)
    for path in plan["suite_files"]:
        print(path, plan["suite_drivers"][path])


if __name__ == "__main__":
    main()
