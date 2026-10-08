#!/usr/bin/env python3
"""Regression detector for CPU suite raw runs (tenferro-rs #1946 B5).

Two modes, never mixed:

``scan``
    Compare one baseline run with one candidate run (usually low-repetition
    ``BENCH_EFFORT=scan`` runs over broad coverage). Timing output is only a
    *suspicion*: ``SUSPECT_SLOWER`` / ``SUSPECT_FASTER`` when the per-op sample
    ranges do not overlap, otherwise ``NO_SIGNAL``. Scan never fails on timing.

``confirm``
    Evaluate a paired directory written by ``scripts/run_paired_timing.sh``:
    baseline/candidate processes run sequentially in balanced (ABBA) order on
    one host with one harness, plus an A/A noise directory. Every threshold,
    the statistic and the repetitions come from a *declared* confirmation
    config (``benchmarks/cpu/confirmation.yaml``); the detector refuses a
    placeholder. Verdicts: ``REGRESSION`` / ``IMPROVEMENT`` / ``NO_CHANGE`` or
    ``INCONCLUSIVE`` (unbalanced or incomplete rounds, noisy arms, missing or
    too-wide A/A noise). Every round's raw samples are used; nothing is
    re-selected.

Deterministic findings are reported in both modes and are independent of
timing noise: ``MISSING`` (expected but not executed), ``NUMERICAL_FAILURE``,
``LIVENESS_FAILURE`` (timeout), ``NEWLY_UNSUPPORTED`` and ``CHANGED_BASELINE``
(different case manifest or harness, or commits that differ from the
declaration).

Exit status: 2 for any deterministic finding, else 1 for a confirmed
REGRESSION, else 0.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_selection  # noqa: E402
import case_status as cs  # noqa: E402

DETERMINISTIC = ("MISSING", "NUMERICAL_FAILURE", "LIVENESS_FAILURE", "NEWLY_UNSUPPORTED",
                 "CHANGED_BASELINE")


# --------------------------------------------------------------------------- loading
def _per_op(row: dict) -> list[float]:
    if "samples_ns" in row:
        ops = row.get("operations_per_sample") or 1
        return [v / ops for v in row["samples_ns"]]
    return [s["elapsed_ns"] / s["iterations"] for s in row.get("samples", [])]


def _key(row: dict) -> str:
    if row.get("case_key"):
        return row["case_key"]
    return cs.case_key(row["case_id"], row["threads"], row.get("provider"))


def load_run(run_dir: Path) -> dict:
    rows = {}
    for path in sorted(run_dir.glob("samples_t*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["per_op"] = _per_op(row)
                rows[_key(row)] = row
    parts = [json.loads(p.read_text()) for p in sorted(run_dir.glob("case_status_t*.json"))]
    status = cs.merge_case_status(parts) if parts else None
    if status is None and (run_dir / "case_status.json").exists():
        status = json.loads((run_dir / "case_status.json").read_text())
    meta = {}
    for path in sorted(run_dir.glob("run_t*.yaml")):
        data = yaml.safe_load(path.read_text())
        meta = {
            "tenferro_commit": data.get("tenferro_rs", {}).get("commit"),
            "tenferro_dirty": data.get("tenferro_rs", {}).get("dirty"),
            "harness_commit": data.get("harness", {}).get("commit"),
            "harness_dirty": data.get("harness", {}).get("dirty"),
            "manifest_version": data.get("collection", {}).get("manifest_version"),
            "hostname": data.get("environment", {}).get("hostname"),
        }
        break
    if status is not None:
        meta.setdefault("manifest_version", status["manifest_version"])
        if meta.get("manifest_version") is None:
            meta["manifest_version"] = status["manifest_version"]
    return {"dir": str(run_dir), "rows": rows, "status": status, "meta": meta}


def deterministic(baseline: dict | None, candidate: dict) -> list[dict]:
    findings = []
    status = candidate["status"]
    if status is None:
        findings.append({"kind": "MISSING", "case": "*", "detail": "candidate has no case_status"})
        return findings
    for key in status["missing"] + status["not_selected"]:
        findings.append({"kind": "MISSING", "case": key,
                         "detail": "expected by the manifest but not executed"})
    for key in status["failed"]:
        row = candidate["rows"].get(key, {})
        kind = "LIVENESS_FAILURE" if row.get("status") == "timeout" else "NUMERICAL_FAILURE"
        findings.append({"kind": kind, "case": key, "detail": str(row.get("error", ""))[:300]})
    if baseline and baseline["status"]:
        before = set(baseline["status"]["executed"])
        for key in status["unsupported"]:
            if key in before:
                findings.append({"kind": "NEWLY_UNSUPPORTED", "case": key,
                                 "detail": candidate["rows"].get(key, {}).get("unsupported_reason", "")})
        bm, cm = baseline["meta"], candidate["meta"]
        if bm.get("manifest_version") != cm.get("manifest_version"):
            findings.append({"kind": "CHANGED_BASELINE", "case": "*",
                             "detail": f"manifest version {bm.get('manifest_version')} vs "
                                       f"{cm.get('manifest_version')}"})
        if bm.get("harness_commit") != cm.get("harness_commit"):
            findings.append({"kind": "CHANGED_BASELINE", "case": "*",
                             "detail": f"harness {bm.get('harness_commit')} vs {cm.get('harness_commit')}"})
        # Two arms may only be compared when each shared case measured the same
        # provider on the same number of workers: everything else is a verdict
        # about a different configuration, not about a code change.
        for key in sorted(set(baseline["rows"]) & set(candidate["rows"])):
            before_row, after_row = baseline["rows"][key], candidate["rows"][key]
            for field in ("provider", "worker_count"):
                if before_row.get(field) != after_row.get(field):
                    findings.append({"kind": "CHANGED_BASELINE", "case": key,
                                     "detail": f"{field} {before_row.get(field)} vs {after_row.get(field)}"})
    return findings


# --------------------------------------------------------------------------- scan
def scan(baseline: dict, candidate: dict) -> list[dict]:
    out = []
    shared = sorted(set(baseline["rows"]) & set(candidate["rows"]))
    for key in shared:
        b, c = baseline["rows"][key], candidate["rows"][key]
        if b.get("status") != "passed" or c.get("status") != "passed" or not b["per_op"] or not c["per_op"]:
            continue
        bmed, cmed = statistics.median(b["per_op"]), statistics.median(c["per_op"])
        if min(c["per_op"]) > max(b["per_op"]):
            label = "SUSPECT_SLOWER"
        elif max(c["per_op"]) < min(b["per_op"]):
            label = "SUSPECT_FASTER"
        else:
            label = "NO_SIGNAL"
        noisy = any((cs.coefficient_of_variation(r["per_op"]) or 0) > cs.NOISY_COV for r in (b, c))
        out.append({"case": key, "label": label, "noisy": noisy, "baseline_median_ns": bmed,
                    "candidate_median_ns": cmed, "ratio": cmed / bmed if bmed else None,
                    "samples": [len(b["per_op"]), len(c["per_op"])]})
    return out


# --------------------------------------------------------------------------- confirm
def load_paired(paired_dir: Path) -> dict:
    plan = json.loads((paired_dir / "plan.json").read_text())
    arms = defaultdict(dict)
    for entry in plan["order"]:
        run_dir = paired_dir / entry["dir"]
        if run_dir.is_dir():
            arms[entry["arm"]][entry["round"]] = load_run(run_dir)
    return {"plan": plan, "arms": arms}


def balanced(order: list[dict], arms: tuple[str, str]) -> bool:
    rounds = defaultdict(list)
    for entry in order:
        rounds[entry["round"]].append(entry["arm"])
    if not rounds or any(sorted(v) != sorted(arms) for v in rounds.values()):
        return False
    firsts = [v[0] for _, v in sorted(rounds.items())]
    return firsts.count(arms[0]) == firsts.count(arms[1])


def round_medians(paired: dict, arm: str, key: str) -> tuple[list[float], list[float]]:
    medians, covs = [], []
    for _, run in sorted(paired["arms"].get(arm, {}).items()):
        row = run["rows"].get(key)
        if row is None or row.get("status") != "passed" or not row["per_op"]:
            return [], []
        medians.append(statistics.median(row["per_op"]))
        covs.append(cs.coefficient_of_variation(row["per_op"]) or 0.0)
    return medians, covs


def paired_statistic(paired: dict, arm_a: str, arm_b: str, key: str):
    a, ca = round_medians(paired, arm_a, key)
    b, cb = round_medians(paired, arm_b, key)
    if not a or len(a) != len(b):
        return None
    ratios = [y / x for x, y in zip(a, b) if x]
    return {"statistic": statistics.median(ratios), "ratios": ratios,
            "a_median": statistics.median(a), "b_median": statistics.median(b),
            "max_cov": max(ca + cb), "rounds": len(ratios)}


def confirm(config: dict, paired: dict, aa: dict | None) -> tuple[list[dict], list[dict]]:
    """Return (comparability findings, per-case verdicts)."""
    findings = []
    plan = paired["plan"]
    rel = float(config["thresholds"]["relative"])
    abs_ns = float(config["thresholds"]["absolute_ns"])
    max_cov = float(config["noise"]["max_cov"])
    max_aa = float(config["noise"]["max_aa_relative_spread"])
    rounds_required = int(config["repetitions"]["rounds"])
    lib = config["library"]
    harnesses = set()
    for arm, expected_commit in (("baseline", lib["baseline_commit"]),
                                 ("candidate", lib["candidate_commit"])):
        for rnd, run in paired["arms"].get(arm, {}).items():
            meta = run["meta"]
            harnesses.add((meta.get("harness_commit"), meta.get("harness_dirty")))
            if not str(meta.get("tenferro_commit", "")).startswith(str(expected_commit)[:12]):
                findings.append({"kind": "CHANGED_BASELINE", "case": f"{arm} round {rnd}",
                                 "detail": f"tenferro-rs {meta.get('tenferro_commit')} != declared "
                                           f"{expected_commit}"})
            if meta.get("manifest_version") != config["cases"]["manifest_version"]:
                findings.append({"kind": "CHANGED_BASELINE", "case": f"{arm} round {rnd}",
                                 "detail": f"manifest {meta.get('manifest_version')} != declared "
                                           f"{config['cases']['manifest_version']}"})
    if len(harnesses) > 1:
        findings.append({"kind": "CHANGED_BASELINE", "case": "*",
                         "detail": f"rounds used different harness revisions {sorted(map(str, harnesses))}"})
    features = {}
    for arm in ("baseline", "candidate"):
        seen = {row.get("cpu_features") for run in paired["arms"].get(arm, {}).values()
                for row in run["rows"].values()}
        seen.discard(None)
        if len(seen) > 1:
            findings.append({"kind": "CHANGED_BASELINE", "case": arm,
                             "detail": f"rounds measured different CPU features {sorted(seen)}"})
        features[arm] = seen
    if features["baseline"] and features["candidate"] and features["baseline"] != features["candidate"]:
        findings.append({"kind": "CHANGED_BASELINE", "case": "*",
                         "detail": f"CPU features {sorted(features['baseline'])} vs "
                                   f"{sorted(features['candidate'])}"})
    for commit, dirty in harnesses:
        if commit and not str(commit).startswith(str(config["harness_commit"])[:12]):
            findings.append({"kind": "CHANGED_BASELINE", "case": "*",
                             "detail": f"harness {commit} != declared {config['harness_commit']}"})
        if dirty:
            findings.append({"kind": "CHANGED_BASELINE", "case": "*",
                             "detail": "harness checkout was dirty during collection"})
    is_balanced = balanced(plan["order"], ("baseline", "candidate"))
    keys = sorted(set().union(*(run["rows"].keys() for arm in paired["arms"].values()
                                for run in arm.values())))
    verdicts = []
    for key in keys:
        stat = paired_statistic(paired, "baseline", "candidate", key)
        verdict = {"case": key}
        if stat is None:
            verdict.update(verdict="INCONCLUSIVE", reason="a round is missing or did not pass")
            verdicts.append(verdict)
            continue
        verdict.update(stat)
        aa_stat = paired_statistic(aa, "a", "b", key) if aa else None
        aa_spread = abs(aa_stat["statistic"] - 1.0) if aa_stat else None
        verdict["aa_spread"] = aa_spread
        diff = stat["b_median"] - stat["a_median"]
        reason = None
        if not is_balanced:
            reason = "baseline/candidate order is not balanced (ABBA)"
        elif stat["rounds"] < rounds_required:
            reason = f"{stat['rounds']} complete rounds < declared {rounds_required}"
        elif stat["max_cov"] > max_cov:
            reason = f"arm CoV {stat['max_cov']:.3f} > declared {max_cov}"
        elif aa_stat is None:
            reason = "no A/A noise characterization for this case"
        elif aa_spread > max_aa:
            reason = f"A/A spread {aa_spread:.3f} > declared {max_aa}"
        elif aa_spread >= rel:
            reason = f"A/A spread {aa_spread:.3f} reaches the relative threshold {rel}"
        if reason:
            verdict.update(verdict="INCONCLUSIVE", reason=reason)
        elif stat["statistic"] > 1 + rel and diff > abs_ns:
            verdict.update(verdict="REGRESSION", reason=f"ratio {stat['statistic']:.3f}, +{diff:.1f} ns/op")
        elif stat["statistic"] < 1 / (1 + rel) and -diff > abs_ns:
            verdict.update(verdict="IMPROVEMENT", reason=f"ratio {stat['statistic']:.3f}, {diff:.1f} ns/op")
        else:
            verdict.update(verdict="NO_CHANGE", reason="within the declared thresholds")
        verdicts.append(verdict)
    return findings, verdicts


def characterize_aa(aa: dict, json_path: Path | None, md_path: Path | None) -> int:
    """Report per-case A/A statistics so thresholds can be declared from them."""
    keys = sorted(set().union(*(run["rows"].keys() for arm in aa["arms"].values()
                                for run in arm.values())))
    rows = []
    for key in keys:
        stat = paired_statistic(aa, "a", "b", key)
        if stat:
            rows.append({"case": key, "aa_statistic": stat["statistic"],
                         "aa_spread": abs(stat["statistic"] - 1), "round_ratios": stat["ratios"],
                         "max_cov": stat["max_cov"], "rounds": stat["rounds"]})
        else:
            rows.append({"case": key, "aa_statistic": None, "reason": "incomplete rounds"})
    lines = ["# A/A noise characterization", "",
             "Same tenferro-rs build in both arms, balanced order. Use these spreads to declare "
             "`noise.max_aa_relative_spread`, `noise.max_cov` and the relative threshold in the "
             "confirmation config **before** any candidate run. No verdicts are produced here.", "",
             f"- Balanced order: {'yes' if balanced(aa['plan']['order'], ('a', 'b')) else 'NO'}", "",
             "| Case | A/A statistic | Spread | Max CoV | Rounds | Round ratios |",
             "|---|---:|---:|---:|---:|---|"]
    for r in rows:
        if r["aa_statistic"] is None:
            lines.append(f"| `{r['case']}` | — | — | — | — | {r['reason']} |")
        else:
            lines.append(f"| `{r['case']}` | {r['aa_statistic']:.4f} | {r['aa_spread']:.4f} | "
                         f"{r['max_cov']:.4f} | {r['rounds']} | "
                         f"{', '.join(f'{x:.4f}' for x in r['round_ratios'])} |")
    text = "\n".join(lines) + "\n"
    if json_path:
        json_path.write_text(json.dumps(rows, indent=2) + "\n")
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(text)
    print(text)
    return 0


# --------------------------------------------------------------------------- report
def render(result: dict) -> str:
    lines = [f"# Regression detection ({result['mode']})", ""]
    if result["mode"] == "scan":
        lines += ["Scan mode: timing labels are **suspicions only**, never performance conclusions. "
                  "Confirm suspects with paired confirmation runs.", ""]
    lines += [f"- Overall: **{result['status']}**", ""]
    det = result["deterministic"]
    lines += ["## Deterministic findings", ""]
    if det:
        lines += ["| Kind | Case | Detail |", "|---|---|---|"]
        lines += [f"| {f['kind']} | `{f['case']}` | {f['detail']} |" for f in det]
    else:
        lines.append("None.")
    lines += ["", "## Timing", ""]
    if result["mode"] == "scan":
        lines += ["| Case | Label | Baseline median ns/op | Candidate median ns/op | Ratio | Noisy | Samples |",
                  "|---|---|---:|---:|---:|---|---|"]
        for t in result["timing"]:
            lines.append(f"| `{t['case']}` | {t['label']} | {t['baseline_median_ns']:.2f} | "
                         f"{t['candidate_median_ns']:.2f} | {t['ratio']:.3f} | {'yes' if t['noisy'] else ''} | "
                         f"{t['samples'][0]}/{t['samples'][1]} |")
    else:
        lines += ["| Case | Verdict | Statistic | Rounds | A/A spread | Reason |",
                  "|---|---|---:|---:|---:|---|"]
        for t in result["timing"]:
            stat = f"{t['statistic']:.3f}" if "statistic" in t else "—"
            aa = f"{t['aa_spread']:.3f}" if t.get("aa_spread") is not None else "—"
            lines.append(f"| `{t['case']}` | {t['verdict']} | {stat} | {t.get('rounds', '—')} | {aa} | {t['reason']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--baseline", type=Path, required=True)
    s.add_argument("--candidate", type=Path, required=True)
    a = sub.add_parser("aa", help="characterize A/A noise (no thresholds, no verdicts)")
    a.add_argument("--aa-dir", type=Path, required=True)
    a.add_argument("--json", type=Path)
    a.add_argument("--markdown", type=Path)
    c = sub.add_parser("confirm")
    c.add_argument("--config", type=Path, required=True)
    c.add_argument("--paired-dir", type=Path, required=True)
    c.add_argument("--aa-dir", type=Path)
    for p in (s, c):
        p.add_argument("--json", type=Path)
        p.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    if args.mode == "aa":
        return characterize_aa(load_paired(args.aa_dir), args.json, args.markdown)
    if args.mode == "scan":
        baseline, candidate = load_run(args.baseline), load_run(args.candidate)
        det = deterministic(baseline, candidate)
        timing = scan(baseline, candidate)
        confirmed_regression = False
    else:
        try:
            config = bench_selection.load_confirmation_config(args.config)
        except bench_selection.SelectionError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        paired = load_paired(args.paired_dir)
        aa = load_paired(args.aa_dir) if args.aa_dir else None
        rounds = sorted(paired["arms"].get("candidate", {}))
        det = []
        if rounds:
            last_candidate = paired["arms"]["candidate"][rounds[-1]]
            last_baseline = paired["arms"].get("baseline", {}).get(rounds[-1])
            det = deterministic(last_baseline, last_candidate)
        comparability, timing = confirm(config, paired, aa)
        det += comparability
        confirmed_regression = any(t["verdict"] == "REGRESSION" for t in timing)
    status = "DETERMINISTIC_FAILURE" if det else ("REGRESSION" if confirmed_regression else "OK")
    result = {"mode": args.mode, "status": status, "deterministic": det, "timing": timing}
    text = render(result)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text)
    print(text)
    return 2 if det else (1 if confirmed_regression else 0)


if __name__ == "__main__":
    raise SystemExit(main())
