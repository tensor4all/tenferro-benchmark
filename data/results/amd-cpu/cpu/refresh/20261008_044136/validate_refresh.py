from pathlib import Path
from collections import Counter
import csv, json
root = Path("data/results/amd-cpu/cpu")
runs = {"einsum": ["20261008_044146", "20261008_053424"], "fft": ["20261008_060740"], "public_api": ["20261008_060822"], "small_work": ["20261008_062614"], "session_matrix": ["20261008_062752"], "route_contract": ["20261008_062904"], "permutation": ["20261008_062910"]}
perf = sorted((root / "perf_issues").glob("20261008_*/report.md"))
if perf:
    runs["perf_issues"] = [perf[-1].parent.name]
summary = {"runs": runs, "csv": {}, "coverage": {}, "route": {}}
for suite, timestamps in runs.items():
    for timestamp in timestamps:
        folder = root / suite / timestamp
        for path in folder.glob("*.csv"):
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                continue
            fields = ("suite", "benchmark", "dtype", "threads", "shape", "backend")
            if all(field in rows[0] for field in fields):
                keys = [tuple(row[field] for field in fields) for row in rows]
                assert len(keys) == len(set(keys)), path
            counts = Counter(row.get("status", "unlabeled") for row in rows)
            assert not any(counts[state] for state in ("error", "failed", "missing")), (path, counts)
            summary["csv"][str(path)] = {"rows": len(rows), "status": dict(counts)}
        status = folder / "case_status.json"
        if status.exists():
            data = json.loads(status.read_text())
            assert data["complete"], status
            assert not data["counts"]["failed"] and not data["counts"]["missing"], status
            assert not data.get("duplicate_rows") and not data.get("unexpected_rows"), status
            summary["coverage"][str(status)] = data["counts"]
        contract = folder / "route_contract.json"
        if contract.exists():
            data = json.loads(contract.read_text())
            counts = Counter(row["verdict"] for row in data["verdicts"])
            assert set(counts) <= {"PASS", "UNSUPPORTED"} and counts["PASS"], counts
            summary["route"][str(contract)] = dict(counts)
Path("data/results/amd-cpu/cpu/refresh/20261008_044136/validation-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
