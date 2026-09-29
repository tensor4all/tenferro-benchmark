#!/usr/bin/env python3
"""Generate the cpu/session_matrix case definitions and quick/full manifests.

`data/instances/session_matrix.json` holds two workload families:

* ``loop_matmul`` / ``loop_solve`` (the original cases): N independent small
  matrices, one allocating public call each, in one shared session.
* One-operation batched routes (tenferro-rs #1946 B2): a single batched
  ``dot_general_read`` / ``dot_general_read_into`` / concrete einsum call, pure
  Hadamard contractions and a three-operand einsum chain, with explicit
  route dimensions, Auto-policy boundary sweeps and policy overrides.

A batched op of batch=1024 and the loop of 1024 independent 4x4 matmuls are
*different workloads* and keep different case IDs; neither stands in for the
other.

The generated manifest (`benchmarks/cpu/manifests/session_matrix.yaml`) is
the versioned expected-case list for the quick and full coverage choices.
Bump MANIFEST_VERSION whenever a case is added, removed or redefined, so the
regression detector refuses to compare runs across incompatible manifests.

Run ``python3 scripts/generate_session_matrix_cases.py`` after editing; the
test suite fails when the checked-in files are stale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
INSTANCES = ROOT / "data/instances/session_matrix.json"
MANIFEST = ROOT / "benchmarks/cpu/manifests/session_matrix.yaml"
SUITE = ROOT / "benchmarks/cpu/session_matrix.yaml"
MANIFEST_VERSION = 2
SUITE_ID = "cpu/session_matrix"

# Default Auto lane cost model of tenferro-cpu (dot_runtime.rs at 5a4e7fd84;
# CpuBatchThresholds defaults in the #1946 repair): one m x n x k item costs
# 50 ns + m*n*k/16 ns and a lane needs >= 8000 ns of estimated work. These
# numbers only place the sweep points around the boundary; they are not
# asserted by any benchmark.
LANE_ITEM_OVERHEAD_NS = 50
LANE_MULADDS_PER_NS = 16
LANE_MIN_WORK_NS = 8000


def auto_lanes(batch: int, m: int, n: int, k: int, threads: int) -> int:
    """Lanes the default cost model predicts (1 = no fan-out)."""
    item = LANE_ITEM_OVERHEAD_NS + (m * n * k) // LANE_MULADDS_PER_NS
    per_lane = max(1, -(-LANE_MIN_WORK_NS // item))
    lanes = min(threads, batch // per_lane)
    return lanes if lanes >= 2 else 1


def loop_cases() -> list[dict]:
    cases = []
    for op in ("matmul", "solve"):
        for n in (2, 4, 8, 16, 32):
            cases.append({
                "id": f"{op}_n{n}_count1024",
                "workload": f"loop_{op}",
                "operation": op,
                "shape": [n, n],
                "dtype": "f64",
                "operations_per_sample": 1024,
                "providers": ["blas", "faer", "pytorch"],
                "coverage": "quick",
                "route": {
                    "execution_path": "concrete",
                    "entrypoint": "BackendSession::dot_general_read" if op == "matmul"
                    else "TensorLinalgExt::solve",
                    "output": "allocating",
                    "representation": "owned",
                    "layout": "direct",
                    "session_boundary": "shared-session",
                    "policy": "not-batched",
                    "workload_shape": "loop-of-1024-independent-ops",
                },
                "intent": (
                    "1024 independent small calls in one session; the distinct-workload "
                    "control for the batch=1024 one-operation cases (tenferro-rs #1946 B2)."
                    if (op, n) == ("matmul", 4) else
                    "Per-call fixed cost of independent small public calls in one session."
                ),
            })
    return cases


POLICY_LABELS = {
    "auto": "auto",
    "backend-outer-min-items-4096": "auto+backend(outer_min_items=4096)",
    "scoped-outer-min-items-4096": "auto+scoped(outer_min_items=4096)",
    "forced-sequential": "forced:Sequential",
    "forced-outer-parallel": "forced:OuterParallel",
    "forced-provider-items": "forced:ProviderItems",
    "forced-whole-batch-vendor": "forced:WholeBatchVendor",
    "backend-lane-min-work-0": "auto+backend(lane_min_work_ns=0)",
    "scoped-lane-min-work-0": "auto+scoped(lane_min_work_ns=0)",
}


def policy_spec(name: str) -> dict:
    if name == "auto":
        return {"strategy": "auto", "override": "none", "thresholds": {}}
    if name.startswith("forced-"):
        strategy = name.removeprefix("forced-").replace("-", "_")
        return {"strategy": strategy, "override": "backend", "thresholds": {}}
    scope, _, knob = name.partition("-")
    thresholds = ({"outer_min_items": 4096} if knob == "outer-min-items-4096"
                  else {"lane_min_work_ns": 0})
    return {"strategy": "auto", "override": scope, "thresholds": thresholds}


def batched_case(*, workload: str, output: str, dtype: str = "f64", batch: int = 1024,
                 m: int = 4, n: int = 4, k: int = 4, layout: str = "direct",
                 policy: str = "auto", coverage: str, intent: str) -> dict:
    op = {"bdot": "dot_general_read", "beinsum": "einsum"}[workload]
    if output == "into":
        op += "_into"
    shape = f"b{batch}_m{m}n{n}k{k}"
    case_id = f"{workload}_{dtype}_{shape}_{layout}_{output}_{policy}"
    pair = f"{workload}_{dtype}_{shape}_{layout}_{policy}"
    representation = "owned" if layout == "direct" else "view"
    entrypoint = {
        ("bdot", "alloc"): "BackendSession::dot_general_read",
        ("bdot", "into"): "BackendSession::dot_general_read_into",
        ("beinsum", "alloc"): "TensorEinsumExt::einsum" if layout == "direct"
        else "TensorReadEinsumExt::einsum_read",
        ("beinsum", "into"): "TensorEinsumIntoExt::einsum_into" if layout == "direct"
        else "TensorReadEinsumIntoExt::einsum_read_into",
    }[(workload, output)]
    return {
        "id": case_id,
        "workload": "batched_dot" if workload == "bdot" else "batched_einsum",
        "operation": op,
        "dtype": dtype,
        "batch": batch,
        "m": m, "n": n, "k": k,
        "subscripts": "ijb,jkb->ikb" if workload == "beinsum" else None,
        "layout": layout,
        "output": output,
        "policy": policy_spec(policy),
        "policy_id": policy,
        "providers": ["faer"],
        "coverage": coverage,
        "pair": pair,
        "requires": (["lane_cost_policy"] if "lane-min-work" in policy else []),
        "predicted_auto_lanes_4t": auto_lanes(batch, m, n, k, 4),
        "route": {
            "execution_path": "concrete-einsum" if workload == "beinsum" else "concrete",
            "entrypoint": entrypoint,
            "output": "allocating" if output == "alloc" else "reused",
            "representation": representation,
            "layout": {"direct": "direct", "strided": "direct-strided",
                       "canonical": "canonical-copy"}[layout],
            "session_boundary": "shared-session",
            "policy": POLICY_LABELS[policy],
            "workload_shape": "one-batched-op",
        },
        "intent": intent,
    }


def hadamard_case(*, n: int, via: str, coverage: str) -> dict:
    op = "dot_general_read" if via == "dot" else "einsum"
    return {
        "id": f"hadamard_f64_m{n}n{n}_{via}_alloc",
        "workload": "hadamard",
        "operation": op,
        "dtype": "f64",
        "batch": 1, "m": n, "n": n, "k": 1,
        "subscripts": "ab,ab->ab" if via == "einsum" else None,
        "layout": "direct",
        "output": "alloc",
        "policy": policy_spec("auto"),
        "policy_id": "auto",
        "providers": ["faer"],
        "coverage": coverage,
        "pair": None,
        "requires": [],
        "route": {
            "execution_path": "concrete-einsum" if via == "einsum" else "concrete",
            "entrypoint": "TensorEinsumExt::einsum" if via == "einsum"
            else "BackendSession::dot_general_read",
            "output": "allocating",
            "representation": "owned",
            "layout": "direct",
            "session_boundary": "shared-session",
            "policy": "auto",
            "workload_shape": "one-batch-only-contraction",
        },
        "intent": ("Pure Hadamard (batch-only) contraction; a lowering that runs one 1x1x1 "
                   "GEMM per element instead of an elementwise multiply shows up here "
                   "(tenferro-rs #1898). The dot_general and einsum spellings expose "
                   "wrapper/lowering divergence."),
    }


def chain_case(*, n: int, coverage: str) -> dict:
    return {
        "id": f"chain3_f64_n{n}_einsum_alloc",
        "workload": "chain3",
        "operation": "einsum",
        "dtype": "f64",
        "batch": 1, "m": n, "n": n, "k": n,
        "subscripts": "ij,jk,kl->il",
        "layout": "direct",
        "output": "alloc",
        "policy": policy_spec("auto"),
        "policy_id": "auto",
        "providers": ["faer"],
        "coverage": coverage,
        "pair": None,
        "requires": [],
        "route": {
            "execution_path": "concrete-einsum",
            "entrypoint": "TensorEinsumExt::einsum",
            "output": "allocating",
            "representation": "owned",
            "layout": "direct",
            "session_boundary": "shared-session",
            "policy": "auto",
            "workload_shape": "multi-operand-contraction",
        },
        "intent": ("Three-operand contraction through the public concrete einsum route: "
                   "path planning plus two binary contractions per call "
                   "(tiny fixed-overhead and larger control)."),
    }


def batched_cases() -> list[dict]:
    cases = []

    def pair(workload: str, coverage: str, intent: str, **kw) -> None:
        for output in ("alloc", "into"):
            cases.append(batched_case(workload=workload, output=output,
                                      coverage=coverage, intent=intent, **kw))

    f2 = ("tenferro-rs #1946 F2 reproducer: one batch=1024 4x4x4 f64 batched dot. At 5a4e7fd84 "
          "the allocating route skipped the Auto lane split that the _into route took.")
    pair("bdot", "quick", f2)
    pair("beinsum", "quick", "Public concrete einsum spelling of the F2 batched contraction "
         "(lowering/wrapper divergence from dot_general).")
    pair("bdot", "quick", "Below the default Auto lane cost cutoff (no fan-out expected).",
         batch=64)
    pair("bdot", "quick", "Larger-matrix control: few items, each above the lane cost cutoff.",
         batch=16, m=64, n=64, k=64)
    pair("bdot", "quick", "Skinny (matrix-vector) batched control.", batch=64, m=64, n=1, k=64)
    pair("bdot", "quick", "Canonical-copy layout: batch-innermost storage leaves neither "
         "matrix axis unit-stride, so the GEMM input needs canonicalization.",
         layout="canonical")
    # Expanded (full) set.
    pair("bdot", "full", "Direct-strided layout: padded batch stride, GEMM-compatible items.",
         layout="strided")
    pair("bdot", "full", "Complex F2 batched dot.", dtype="c64")
    pair("bdot", "full", "Complex non-contiguous batched dot.", dtype="c64", layout="strided")
    pair("beinsum", "full", "Einsum over non-contiguous (padded-batch) views.", layout="strided")
    for batch in (256, 320, 4096):
        pair("bdot", "full", f"Auto batch-size sweep around the lane cost boundary "
             f"(default model predicts {auto_lanes(batch, 4, 4, 4, 4)} lanes at 4T).",
             batch=batch)
    for size in (8, 16, 32):
        pair("bdot", "full", f"Auto matrix-size sweep at batch 64 (default model predicts "
             f"{auto_lanes(64, size, size, size, 4)} lanes at 4T).",
             batch=64, m=size, n=size, k=size)
    for policy in ("backend-outer-min-items-4096", "scoped-outer-min-items-4096"):
        pair("bdot", "full", "Threshold override suppressing fan-out; backend default vs "
             "scoped precedence.", policy=policy)
    for policy in ("forced-sequential", "forced-outer-parallel", "forced-provider-items",
                   "forced-whole-batch-vendor"):
        pair("bdot", "full", "Forced batch strategy; a provider/route that cannot honor it "
             "must fail typed (recorded unsupported), never fall back.", policy=policy)
    for policy in ("backend-lane-min-work-0", "scoped-lane-min-work-0"):
        pair("bdot", "full", "Lane cost model override (tenferro-rs #1946 F3) below the "
             "default cutoff; unsupported where the revision hard-codes the cost model.",
             batch=64, policy=policy)
    cases.append(hadamard_case(n=64, via="dot", coverage="quick"))
    cases.append(hadamard_case(n=64, via="einsum", coverage="quick"))
    cases.append(hadamard_case(n=256, via="dot", coverage="full"))
    cases.append(hadamard_case(n=256, via="einsum", coverage="full"))
    cases.append(chain_case(n=4, coverage="quick"))
    cases.append(chain_case(n=64, coverage="full"))
    return cases


def suite_yaml(ids: list[str]) -> str:
    suite = {
        "schema_version": 1,
        "suite_id": SUITE_ID,
        "title": "Many small matrices and one-operation batched routes in one CPU session",
        "description": (
            "1024 independent f64 matmul or solve calls, and one-operation batched "
            "dot_general_read/_into, concrete einsum, Hadamard and three-operand routes "
            "(tenferro-rs #1946 B2), with all inputs and a shared session prepared before "
            "timing. Coverage (quick/full) comes from benchmarks/cpu/manifests/"
            "session_matrix.yaml; effort (scan/standard/confirm) is chosen separately."),
        "defaults": {
            "device": {"kind": "cpu", "ordinal": 0},
            "run": {"warmups": 3, "runs": 15, "min_runtime_ms": 1,
                    "timing_scope": "steady_state_host_api"},
            "effort": {"scan": {"warmups": 1, "runs": 3}},
            "verify": {"reference": "analytic matmul, solve residual and independent "
                                    "contraction references", "rtol": 1.0e-11, "atol": 1.0e-11},
        },
        "backends": ["tenferro-blas-shared-session", "tenferro-faer-shared-session",
                     "pytorch-python-loop"],
        "problems": {"source": "data/instances/session_matrix.json", "include": ids},
    }
    return ("# Generated by scripts/generate_session_matrix_cases.py; do not edit.\n"
            + yaml.safe_dump(suite, sort_keys=False, width=100))


def generate() -> tuple[str, str, str]:
    cases = loop_cases() + batched_cases()
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate case IDs")
    instances = json.dumps(cases, indent=2) + "\n"
    quick = [c["id"] for c in cases if c["coverage"] == "quick"]
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "suite_id": SUITE_ID,
        "source": str(INSTANCES.relative_to(ROOT)),
        "generated_by": "scripts/generate_session_matrix_cases.py",
        "coverage": {"quick": quick, "full": ids},
        "route_contract_pairs": sorted({c["pair"] for c in cases if c.get("pair")}),
    }
    header = ("# Generated by scripts/generate_session_matrix_cases.py; do not edit.\n"
              "# quick is the routine coverage choice (F2 reproducer plus controls);\n"
              "# full is the selectable expanded matrix. Measurement effort (scan/standard/\n"
              "# confirm) is chosen independently; see docs/regression-detection.md.\n")
    return instances, header + yaml.safe_dump(manifest, sort_keys=False), suite_yaml(ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if checked-in files are stale")
    args = parser.parse_args()
    instances, manifest, suite = generate()
    if args.check:
        stale = [p for p, text in ((INSTANCES, instances), (MANIFEST, manifest), (SUITE, suite))
                 if not p.exists() or p.read_text() != text]
        for path in stale:
            print(f"stale: {path.relative_to(ROOT)}", file=sys.stderr)
        return 1 if stale else 0
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    INSTANCES.write_text(instances)
    MANIFEST.write_text(manifest)
    SUITE.write_text(suite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
