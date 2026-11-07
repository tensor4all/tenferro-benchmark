"""Materialize and account for the single routine CUDA suite set."""
import argparse
import copy
import json
import math
import statistics
import sys
from pathlib import Path

import yaml

import bench_selection
import report_summary

DEFAULT_DRIVER = "gpu_suite"
MISSING_REASON = report_summary.MISSING_REASON


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


def suite_problems(root, spec):
    """The suite declaration and its selectable cases, for either problem form.

    A suite names its cases either as a `problems` list or as
    `problems: {source, include}` over an instance file; `instances` is the
    instance document in the second form, so the routine selection can hand the
    collectors exactly the cases it selected.
    """
    source = yaml.safe_load((root/spec["file"]).read_text())
    problems = source["problems"]
    if isinstance(problems, list):
        return source, {p["id"]: p for p in problems}, None
    instances = yaml.safe_load((root/problems["source"]).read_text())
    declared = set(problems.get("include") or [])
    # Instance documents name their list `cases` (public API) or `patterns`
    # (permutation); both carry the same per-entry `id`.
    entries = instances.get("cases") if "cases" in instances else instances["patterns"]
    cases = {case["id"]: case for case in entries}
    if not set(spec["problems"]) <= declared or not set(spec["problems"]) <= cases.keys():
        raise ValueError(f"stale routine problem manifest: {spec['file']}")
    return source, cases, instances


def suite_payload(instances, source, key, selected, backends):
    """The suite document a collector reads: cases, identity and the page note.

    Both selection paths (the per-family command and the routine manifest) hand
    the collector the same shape, so the page a run publishes states the
    declaration's own identity and shape class whichever path collected it.
    """
    return dict(
        instances,
        suite_id=source["suite_id"],
        backends=backends,
        description=source.get("description", ""),
        **{key: [copy.deepcopy(entry) for entry in selected]},
    )


def instances_for(declaration, output):
    """Write the instance document one suite declaration selects.

    The routine selection materializes a document for its curated case set; the
    per-family command needs the same document for the whole declaration.  Both
    derive it from the declaration's ``problems.include`` over its instance
    document, and both stamp the declaration's own identity on it.
    """
    source = yaml.safe_load(Path(declaration).read_text())
    problems = source["problems"]
    if not isinstance(problems, dict):
        raise ValueError(f"{declaration}: problems is not an instance selection")
    instances = yaml.safe_load(Path(problems["source"]).read_text())
    key = "cases" if "cases" in instances else "patterns"
    entries = {entry["id"]: entry for entry in instances[key]}
    missing = [problem for problem in problems["include"] if problem not in entries]
    if missing:
        raise ValueError(f"{declaration}: cases absent from {problems['source']}: {missing}")
    payload = suite_payload(
        instances,
        source,
        key,
        [entries[problem] for problem in problems["include"]],
        source["backends"],
    )
    Path(output).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def materialize(manifest_path, requested, output, backend_override="", problem="", effort="standard",
                vendor_arms=False):
    manifest = yaml.safe_load(manifest_path.read_text())
    root = manifest_path.resolve().parents[3]
    output.mkdir(parents=True, exist_ok=True)
    expected, selected, files, drivers = [], [], [], {}
    instances_files = {}
    requested = {(Path(path).resolve().relative_to(root).as_posix() if Path(path).is_absolute()
                  else Path(path).as_posix()) for path in requested}
    known_files = {spec["file"] for spec in manifest["suites"].values()}
    if requested - known_files:
        raise ValueError(f"suites absent from routine manifest: {sorted(requested-known_files)}")
    for suite_id, spec in manifest["suites"].items():
        source, problems, instances = suite_problems(root, spec)
        if not set(spec["problems"]) <= problems.keys():
            raise ValueError(f"stale routine problem manifest: {suite_id}")
        backends = list(spec["backends"])
        if vendor_arms:
            backends += list(spec.get("vendor_backends") or [])
        if not set(backends) <= set(source["backends"]):
            raise ValueError(f"routine backends absent from {spec['file']}: {suite_id}")
        if backend_override:
            backends = [b for b in backend_override.split(",") if b in source["backends"]]

        def arms_for(pid):
            """The arms one case declares; every routine backend when it declares none.

            An instance-driven case may carry ``participants_gpu``, and its driver
            then emits a row for those arms only. Expanding the cartesian product
            here turned every non-participant pair into a `missing` row the
            collector can never produce (#368). The filter keeps ``backends``
            order, so a plan's ids stay stable against the recorded runs.
            """
            participants = (problems[pid] or {}).get("participants_gpu")
            return [b for b in backends if b in participants] if participants else backends

        expected.extend(f"{suite_id}:{pid}:{backend}" for pid in spec["problems"]
                        for backend in arms_for(pid))
        if spec["file"] not in requested or not backends:
            continue
        ids = [pid for pid in spec["problems"] if not problem or pid == problem]
        if not ids:
            continue
        selected.extend(f"{suite_id}:{pid}:{backend}" for pid in ids for backend in arms_for(pid))
        effective = copy.deepcopy(source)
        effective["backends"] = backends
        path = output / (suite_id.replace("/", "_")+".yaml")
        instances_path = None
        if instances is None:
            effective["problems"] = [copy.deepcopy(problems[pid]) for pid in ids]
            for item in effective["problems"]:
                run = dict(source.get("defaults", {}).get("run", {}), **item.get("run", {}))
                resolved = bench_selection.resolve_effort(run, value=effort)
                run.update(warmups=resolved["warmups"], runs=resolved["runs"])
                item["run"] = run
        else:
            # Instance-driven suites read their cases from data/instances, so the
            # selection hands the collector a materialized instance document. It
            # carries the selected suite's own identity, so the page and raw-run
            # path name this suite rather than the instance file's origin.
            key = "cases" if "cases" in instances else "patterns"
            payload = suite_payload(
                instances, source, key, [problems[pid] for pid in ids], backends
            )
            instances_path = output / (suite_id.replace("/", "_") + "_instances.json")
            instances_path.write_text(json.dumps(payload, indent=2) + "\n")
            instances_files[path.as_posix()] = instances_path.as_posix()
            effective["problems"] = {"source": instances_path.as_posix(), "include": ids}
        path.write_text(yaml.safe_dump(effective, sort_keys=False, width=100))
        files.append(path.as_posix())
        drivers[path.as_posix()] = spec.get("driver") or DEFAULT_DRIVER
    if not selected:
        raise ValueError("no GPU cases selected")
    plan = dict(manifest_version=manifest["manifest_version"], effort=effort,
                expected=expected, selected=selected, suite_files=files, suite_drivers=drivers,
                suite_instances=instances_files,
                backend_override=backend_override or None, vendor_arms=vendor_arms,
                problem_filter=problem or None)
    (output/"selection.json").write_text(json.dumps(plan,indent=2)+"\n")
    return plan


def account(plan, paths):
    by_id, duplicates = {}, []
    duplicate_counts = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            key = f"{row['suite_id']}:{row['problem_id']}:{row['backend']}"
            if key in by_id:
                duplicates.append(key)
                duplicate_counts[key] = duplicate_counts.get(key, 1) + 1
            by_id[key] = row
    result = {name:[] for name in ("executed","skipped","unsupported","failed","missing","noisy","verification_only","raw_samples_unavailable")}
    reasons = {}
    statuses = {}
    for key in plan["selected"]:
        row = by_id.get(key)
        if row is None:
            # The case was planned and never measured. It stays in the
            # accounting as `missing` with the reason no row exists, so a
            # reader sees the gap and its cause instead of nothing at all.
            result["missing"].append(key)
            statuses[key] = "missing"
            reasons[key] = MISSING_REASON
            continue
        statuses[key] = row["status"]
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
            result["raw_samples_unavailable"].append(key)
            reasons[key] = "the record passed but carries no sampling batch_ms, so no median can be checked"
            continue
        if not samples:
            result["raw_samples_unavailable"].append(key)
            reasons[key] = "the record carries an empty sampling batch, so no median can be checked"
            continue
        if any(not math.isfinite(value) or value <= 0 for value in samples):
            result["failed"].append(key); continue
        result["executed"].append(key)
        if len(samples)>1 and statistics.stdev(samples)/statistics.mean(samples)>.10:
            result["noisy"].append(key)
    for key in duplicates:
        reasons[key] = (f"{duplicate_counts.get(key, 2)} rows were recorded for this case; "
                        "only one row per case may publish")
    for key in sorted(set(by_id) - set(plan["selected"])):
        statuses[key] = by_id[key]["status"]
        reasons[key] = ("a row was recorded for a case the routine selection did not select; "
                        "it is not part of this campaign")
    result.update(manifest_version=plan["manifest_version"],effort=plan["effort"],
                  expected=plan["expected"],selected=plan["selected"],
                  duplicates=duplicates,unexpected=sorted(set(by_id)-set(plan["selected"])),
                  reasons=reasons, statuses=statuses)
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
    parser.add_argument("--instances",type=Path,metavar="DECLARATION",
        help="write the instance document this suite declaration selects, then exit")
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
    if args.instances is not None:
        instances_for(args.instances, args.output)
        return
    if args.records is not None:
        plan = json.loads((args.output/"selection.json").read_text())
        status = account(plan,args.records)
        (args.output/"case_status.json").write_text(json.dumps(status,indent=2)+"\n")
        for category in ("failed", "skipped", "unsupported", "missing", "duplicates", "unexpected"):
            print(f"{category}: {status['counts'][category]}")
            for key in status[category]:
                print(f"  {key}: {status['reasons'].get(key, category)}")
        # case_status.json is already written: the rows this run did measure stay
        # publishable, and the non-zero exit only tells the job it went red.
        incomplete = [name for name in ("failed", "missing", "duplicates", "unexpected")
                      if status[name]]
        if incomplete:
            print("collection incomplete: " + ", ".join(
                f"{name}={status['counts'][name]}" for name in incomplete), file=sys.stderr)
            raise SystemExit("GPU collection has failed/missing/duplicate/unexpected cases")
        return
    plan = materialize(args.manifest,args.suites,args.output,args.backends,args.problem,args.effort,
                       args.vendor_arms)
    for path in plan["suite_files"]:
        print(path, plan["suite_drivers"][path], plan["suite_instances"].get(path, "-"))


if __name__ == "__main__":
    main()
