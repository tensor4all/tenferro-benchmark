#!/usr/bin/env python3
"""Build a tprims-rs benchmark corpus from TPRIMS_SHAPE_LOG files.

A `--features tprims` build of this suite run with TPRIMS_SHAPE_LOG=<file>
appends one JSON line per tprims provider call (src/cpu_provider.rs). This
script groups the executed calls by shape and layout, ranks the groups by
total time, and writes the corpus format of tprims-rs
(`benchmarks/src/corpus.rs`): `dot_general` entries for contractions and
`gemm_batched` entries for GEMM-slot calls, each with its call count and
share of the logged time.

    tprims_shape_corpus.py LOG... -o corpus.json [--time-share 0.95] [--frequent 20]

Kept: the most expensive groups up to `--time-share` of the logged time, plus
the `--frequent` most frequent groups not already kept.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

DTYPES = {"F32": "f32", "F64": "f64", "C32": "c32", "C64": "c64"}


def col_major(dims: list[int]) -> list[int]:
    strides, acc = [], 1
    for d in dims:
        strides.append(acc)
        acc *= max(d, 1)
    return strides


def dot_general_entry(r: dict) -> dict:
    a, b = r["a"]["dims"], r["b"]["dims"]
    free_a = [d for i, d in enumerate(a) if i not in r["lc"] and i not in r["lb"]]
    free_b = [d for i, d in enumerate(b) if i not in r["rc"] and i not in r["rb"]]
    out = free_a + free_b + [a[i] for i in r["lb"]]
    return {
        "op": "dot_general",
        "dtype": DTYPES[r["dtype"]],
        "a": r["a"],
        "b": r["b"],
        # tenferro allocates contraction outputs compact column-major.
        "c": {"dims": out, "strides": col_major(out)},
        "lc": r["lc"], "rc": r["rc"], "lb": r["lb"], "rb": r["rb"],
        "conj": r.get("conj", [False, False]),
    }


def gemm_entry(r: dict) -> dict:
    m, n, k, batch = r["m"], r["n"], r["k"], r["batch"]
    return {
        "op": "gemm_batched",
        "dtype": DTYPES[r["dtype"]],
        "m": m, "n": n, "k": k, "batch": batch,
        "a": {"dims": [m, k, batch], "strides": r["a"]},
        "b": {"dims": [k, n, batch], "strides": r["b"]},
        "c": {"dims": [m, n, batch], "strides": r["c"]},
    }


def entry_of(r: dict) -> dict | None:
    if r.get("outcome") != "executed" or r.get("dtype") not in DTYPES:
        return None
    if r["op"] == "dot_general":
        return dot_general_entry(r)
    if r["op"] in ("gemm", "gemm_batched"):
        return gemm_entry(r)
    return None  # grouped jobs are replayed by tprims' own grouped rows


def build(records: list[dict], time_share: float, frequent: int) -> list[dict]:
    groups: dict[str, dict] = {}
    calls: dict[str, int] = defaultdict(int)
    ns: dict[str, int] = defaultdict(int)
    for r in records:
        e = entry_of(r)
        if e is None:
            continue
        key = json.dumps(e, sort_keys=True)
        groups[key] = e
        calls[key] += 1
        ns[key] += int(r.get("ns", 0))
    total = sum(ns.values()) or 1
    by_time = sorted(groups, key=lambda k: (-ns[k], k))
    kept, acc = [], 0
    for k in by_time:
        if acc / total >= time_share:
            break
        kept.append(k)
        acc += ns[k]
    for k in sorted(groups, key=lambda k: (-calls[k], k)):
        if frequent <= 0:
            break
        if k not in kept:
            kept.append(k)
            frequent -= 1
    out = []
    for i, k in enumerate(kept):
        e = dict(groups[k])
        e["name"] = f"{e['op']}_{i:03d}_{e['dtype']}"
        e["calls"] = calls[k]
        e["time_share"] = round(ns[k] / total, 6)
        out.append(e)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--time-share", type=float, default=0.95)
    ap.add_argument("--frequent", type=int, default=20)
    ap.add_argument("--source", default="{}", help="JSON object recorded as the corpus source")
    a = ap.parse_args(argv)
    records = []
    for p in a.logs:
        for line in p.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    entries = build(records, a.time_share, a.frequent)
    source = json.loads(a.source)
    source.update({"tool": "tenferro-benchmark scripts/tprims_shape_corpus.py",
                   "logs": [str(p) for p in a.logs], "calls": len(records),
                   "time_share": a.time_share, "frequent": a.frequent})
    a.output.write_text(json.dumps({"source": source, "entries": entries}, indent=1) + "\n")
    print(f"{len(entries)} entries from {len(records)} calls -> {a.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
