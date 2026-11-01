"""Select and account for the routine public CUDA latency/state controls."""
import argparse
import json
import math
from pathlib import Path
import statistics

import yaml

ROOT = Path(__file__).resolve().parents[1]


def plan(effort):
    manifest = yaml.safe_load((ROOT / "benchmarks/gpu/manifests/routine_controls.yaml").read_text())
    cases = {row["id"]: row for row in json.loads((ROOT / manifest["source"]).read_text())}
    selected = {"timing": manifest["timing"], "state": manifest["state"]}
    for lane, keys in selected.items():
        if len(keys) != len(set(keys)) or not set(keys) <= cases.keys():
            raise ValueError("duplicate or unknown routine control")
        for key in keys:
            state = cases[key]["measurement_scope"] in {"state_diagnostic", "counter_diagnostic"}
            if state != (lane == "state"):
                raise ValueError(f"control in wrong measurement lane: {key}")
    return dict(manifest_version=manifest["manifest_version"], effort=effort, **selected)


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
    status = dict(selection, expected=expected,
                  measured=[], observed=[], verification_only=[], missing=[], failed=[], noisy=[],
                  duplicates=duplicates, unexpected=sorted(set(rows) - set(expected)), provenance=provenance,
                  timing_statistics={}, state_statistics={}, sequences={})
    revisions = {(item["tenferro_rs"]["commit"], tuple(item["tenferro_rs"]["features"]),
                  item["harness"]["commit"]) for item in provenance}
    status["provenance_mismatch"] = len(revisions) != 1
    for key in expected:
        row = rows.get(key)
        if row is None:
            status["missing"].append(key)
            continue
        samples = row.get("samples", [])
        if row.get("kind") == "shape_stream":
            status["sequences"][key] = dict(cache_mode=row["cache_mode"],
                                            reset=row["cache_reset_scope"], sequence=row["sequence"])
        if row.get("correctness_status") != "passed":
            status["failed"].append(key)
        elif key in selection["state"]:
            status["failed" if samples else "observed"].append(key)
            if not samples and row.get("kind") == "shape_stream":
                # tenferro-gpu v0.8.0 removed the cuTENSOR plan/workspace counters
                # from the public surface; these are the runtime's own counters.
                snapshots = row["counter_observations"]
                first, last = snapshots[0]["stats"], snapshots[-1]["stats"]
                status["state_statistics"][key] = dict(
                    live_bytes_before=first["device"]["live_bytes"],
                    live_bytes_after=last["device"]["live_bytes"],
                    allocations=last["retirement"]["allocations"]-first["retirement"]["allocations"],
                    launches=last["retirement"]["launches"]-first["retirement"]["launches"],
                    zero_fills=last["retirement"]["zero_fills"]-first["retirement"]["zero_fills"],
                    artifact_entries=last["artifact_cache"]["entries"],
                    module_entries=last["module_cache"]["entries"],
                    observed_pending_bytes_peak=max(item["stats"]["retirement"]["pending_bytes"]
                                                    for item in snapshots))
            elif not samples and row.get("kind") == "vendor_churn":
                status["state_statistics"][key] = {field: row[field] for field in (
                    "cycles", "rss_bytes_before", "rss_bytes_after", "rss_growth_bytes",
                    "free_bytes_before", "free_bytes_after")}
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
            quartiles = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]]*3
            enqueue = row.get("host_enqueue_samples_ns")
            status["timing_statistics"][key] = dict(
                median_ns=statistics.median(values), iqr_ns=quartiles[2]-quartiles[0], samples=len(samples),
                unit="sequence" if row.get("kind") == "shape_stream" else "allocation",
                scope=row.get("measurement_scope"),
                operations_per_sequence=row.get("operations_per_sequence"),
                host_enqueue_median_ns=statistics.median(enqueue) if enqueue else None)
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
             f"Manifest {status['manifest_version']}; effort {status['effort']}.", "",
             "Timing controls measured and state controls observed: " + ("yes" if status["complete"] else "no") + ".", "",
             "State/counter controls contain no timed samples. Verification-only cases "
             "are not measured coverage. These controls do not establish a paired regression conclusion.", "",
             f"Revision/features mismatch across lanes: {status['provenance_mismatch']}.", ""]
    for source in status["provenance"]:
        lines += [f"Raw run: `{source['raw_run']}`; tenferro-rs `{source['tenferro_rs']['commit']}`; "
                  f"harness `{source['harness']['commit']}`; GPU UUID `{source['cuda']['device_uuid']}`.", ""]
    lines += ["| Classification | Count |", "|---|---:|"]
    lines += [f"| {key} | {count} |" for key, count in status["counts"].items()]
    lines += ["", "## Latency observations", "",
              "Times are per declared unit, including host dispatch and native completion synchronization. "
              "Whole-sequence samples include required planning; shared allocation and per-call session entry "
              "are distinct scopes. Host enqueue is not device kernel time. Setup and validation are untimed.", "",
              "| Case | Scope | Unit | Calls/sequence | Samples | Median µs | IQR µs | Host enqueue µs |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for key, value in status["timing_statistics"].items():
        enqueue = value["host_enqueue_median_ns"]
        lines.append(f"| {key} | {value['scope']} | {value['unit']} | "
                     f"{value['operations_per_sequence'] or '—'} | {value['samples']} | "
                     f"{value['median_ns']/1000:.3f} | {value['iqr_ns']/1000:.3f} | "
                     f"{enqueue/1000 if enqueue is not None else '—'} |")
    lines += ["", "## Untimed state observations", "",
              "Live bytes and pool reservations describe public runtime retention, not the whole allocator. "
              "A plan-cache entry/hit/miss/byte count is unavailable after tenferro-gpu v0.8.0's "
              "kernel-catalogue cutover (the cuTENSOR plan cache is private to the engine's handle); "
              "nothing is substituted for it. The pending-byte peak uses snapshots and can miss "
              "activity between them. RSS/free-memory changes alone do not establish a leak.", "",
              "| Case | Public observation |", "|---|---|"]
    for key, value in status["state_statistics"].items():
        lines.append(f"| {key} | `{json.dumps(value, sort_keys=True)}` |")
    lines += ["", "## Sequence/cache contracts", "",
              "Key order is exact. Fixture definitions retain shapes and strides; full per-command "
              "enqueue/completion/drop snapshots remain in the referenced raw runs.", ""]
    for key, value in status["sequences"].items():
        definitions = {json.dumps(item, sort_keys=True) for item in value["sequence"]}
        lines += [f"- `{key}`: mode `{value['cache_mode']}`; reset `{value['reset']}`; "
                  f"order `{[item['key'] for item in value['sequence']]}`; "
                  f"fixtures `{' '.join(sorted(definitions))}`."]
    for key in ("expected", "measured", "observed", "verification_only", "failed", "missing",
                "noisy", "duplicates", "unexpected"):
        lines += ["", f"## {key}", ""] + ([f"- `{case}`" for case in status[key]] or ["None."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
        selection = plan(args.effort)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        if args.lane:
            print(",".join(selection[args.lane]))


if __name__ == "__main__":
    main()
