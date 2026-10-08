#!/usr/bin/env python3
"""Public-API entrypoint and route coverage for benchmarks/cpu/public_api_coverage.yaml.

Two checks (tenferro-rs #1946 B1):

``entrypoints``
    Every ``Type::method`` / ``Type::{a,b}`` / ``Type`` spelling in the
    manifest (``coverage[].tenferro_apis`` and ``routes[].entrypoint``) must
    exist in the tenferro-rs source the harness builds against: the type,
    trait or function is defined, and each method is defined in a file that
    defines or implements that type. Removed spellings fail deterministically.

``routes``
    A route is covered only when every case it names *executed and passed* in
    the given raw runs (their ``case_status.json``). Missing, failed,
    unsupported or unselected cases leave the route uncovered, with the
    reason. Kernel ``alias`` families establish operation coverage only and
    never count as latency/API-route coverage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/cpu/public_api_coverage.yaml"
TENFERRO = ROOT / "extern/tenferro-rs"
SPELLING = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:::(\{[^}]*\}|[a-z_][a-z0-9_]*))?(\s*\(.*\))?$")
DEFINES = re.compile(r"\b(?:trait|struct|enum|type|fn|mod)\s+([A-Za-z_][A-Za-z0-9_]*)")
FN = re.compile(r"\bfn\s+([a-z_][a-z0-9_]*)")


def parse_spelling(text: str) -> tuple[str, list[str]] | None:
    """Return (type, methods) or None for free text."""
    match = SPELLING.match(text.strip())
    if not match:
        return None
    name, methods, _ = match.groups()
    if not methods:
        return name, []
    if methods.startswith("{"):
        return name, [m.strip() for m in methods[1:-1].split(",") if m.strip()]
    return name, [methods]


def _blocks(text: str):
    """Yield (header, body) for every trait/impl block in a Rust source text."""
    for match in re.finditer(r"\b(pub(?:\([^)]*\))?\s+)?(trait|impl)\b", text):
        start = match.end()
        brace = text.find("{", start)
        semi = text.find(";", start)
        if brace < 0 or (0 <= semi < brace):
            continue
        header = text[start:brace]
        depth, j = 1, brace + 1
        while depth and j < len(text):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        yield match.group(2), header, text[brace + 1:j - 1]


def _last_ident(path: str) -> str | None:
    path = re.sub(r"<.*", "", path.strip())  # drop generics
    names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", path)
    return names[-1] if names else None


class SourceIndex:
    """Names defined in tenferro-rs and the methods callable through each."""

    def __init__(self, root: Path):
        self.defined: set[str] = set()
        self.methods: dict[str, set[str]] = defaultdict(set)
        files = sorted(root.glob("crates/*/src/**/*.rs"))
        if not files:
            raise SystemExit(f"no tenferro-rs sources under {root}")
        for path in files:
            text = path.read_text(errors="replace")
            self.defined |= set(DEFINES.findall(text))
            for kind, header, body in _blocks(text):
                fns = set(FN.findall(body))
                if kind == "trait":
                    name = _last_ident(header.split(":")[0])
                    if name:
                        self.methods[name] |= fns
                    continue
                header = re.sub(r"^\s*<[^{]*?>\s*", "", header)  # impl<...>
                header = header.split(" where ")[0]
                if " for " in f" {header} ":
                    trait, self_ty = header.split(" for ", 1)
                    for name in (_last_ident(trait), _last_ident(self_ty)):
                        if name:
                            self.methods[name] |= fns
                else:
                    name = _last_ident(header)
                    if name:
                        self.methods[name] |= fns

    def check(self, name: str, methods: list[str]) -> list[str]:
        if name not in self.defined:
            return [f"{name} is not defined"]
        return [f"{name}::{m} is not defined" for m in methods if m not in self.methods[name]]


def entrypoints(manifest: dict) -> list[tuple[str, str]]:
    """(context, spelling) pairs from the manifest."""
    out = []
    for family in manifest.get("coverage", []):
        for api in family.get("tenferro_apis") or []:
            out.append((family["family"], api))
    for route in manifest.get("routes", []):
        out.append((route["route_id"], route["entrypoint"]))
    return out


def check_entrypoints(manifest: dict, index: SourceIndex) -> list[str]:
    problems = []
    for context, api in entrypoints(manifest):
        parsed = parse_spelling(api)
        if parsed is None:
            family = next((f for f in manifest.get("coverage", []) if f["family"] == context), {})
            if family.get("disposition") != "out_of_scope":
                problems.append(f"{context}: free-text entrypoint {api!r} is not a real spelling")
            continue
        problems += [f"{context}: {p}" for p in index.check(*parsed)]
    return problems


def load_status(run_dirs: list[Path]) -> dict[str, dict[str, list[str]]]:
    """suite_id -> classification -> case keys, merged over runs."""
    merged: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for run_dir in run_dirs:
        path = run_dir / "case_status.json"
        if not path.exists():
            parts = sorted(run_dir.glob("case_status_t*.json"))
            if not parts:
                raise SystemExit(f"{run_dir}: no case_status.json")
            statuses = [json.loads(p.read_text()) for p in parts]
        else:
            statuses = [json.loads(path.read_text())]
        for status in statuses:
            for field in ("selected", "executed", "unsupported", "failed", "missing"):
                merged[status["suite_id"]][field] += status[field]
    return merged


def _base(key: str) -> str:
    return key.split("@t")[0].split("[")[0]


def route_coverage(manifest: dict, status: dict) -> list[dict]:
    rows = []
    for route in manifest.get("routes", []):
        reasons, covered_cases = [], 0
        for ref in route["cases"]:
            suite_id, _, case_id = ref.partition(":")
            if case_id.endswith("]"):
                case_id, _, provider = case_id[:-1].partition("[")
            else:
                provider = None
            s = status.get(suite_id, {})

            def keys(field, cid=case_id, prov=provider):
                return [k for k in s.get(field, []) if _base(k) == cid
                        and (prov is None or f"[{prov}]" in k)]
            if keys("failed"):
                reasons.append(f"{ref} failed")
            elif keys("unsupported"):
                reasons.append(f"{ref} unsupported")
            elif keys("missing"):
                reasons.append(f"{ref} missing")
            elif not keys("executed"):
                reasons.append(f"{ref} not run")
            else:
                covered_cases += 1
        rows.append({"route_id": route["route_id"], "entrypoint": route["entrypoint"],
                     "covered": not reasons and covered_cases == len(route["cases"]),
                     "reasons": reasons})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("entrypoints")
    e.add_argument("--manifest", type=Path, default=MANIFEST)
    e.add_argument("--tenferro-dir", type=Path, default=TENFERRO)
    r = sub.add_parser("routes")
    r.add_argument("--manifest", type=Path, default=MANIFEST)
    r.add_argument("--runs", type=Path, nargs="+", required=True)
    r.add_argument("--json", type=Path)
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    if args.command == "entrypoints":
        problems = check_entrypoints(manifest, SourceIndex(args.tenferro_dir))
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"{len(entrypoints(manifest))} spellings checked, {len(problems)} problems")
        return 1 if problems else 0
    rows = route_coverage(manifest, load_status(args.runs))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    print("| Route | Entrypoint | Covered | Why not |\n|---|---|---|---|")
    for row in rows:
        print(f"| `{row['route_id']}` | `{row['entrypoint']}` | {'yes' if row['covered'] else 'NO'} | "
              f"{'; '.join(row['reasons'])} |")
    return 0 if all(row["covered"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
