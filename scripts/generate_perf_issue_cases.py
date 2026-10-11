#!/usr/bin/env python3
"""Generate the cpu/perf_issues case definitions, suite and quick/full manifest.

Every case reproduces the workload of an open tenferro-rs performance issue
and records that issue number next to it (``issues``), together with its
reference arm(s) in the same family. The policy (tenferro-rs #2010, AGENTS.md)
is that a performance issue gets its workload added here when the issue is
opened; a performance-fix PR reports numbers from these cases.

Rust cases run in ``perf_issue_case`` (public tenferro-rs constructors only:
``CpuBackend::new()`` / ``CpuBackend::with_threads(n)``, ``with_backend_session``
and public concrete, eager and traced ops). ``pytorch-cpu`` cases run inside
``scripts/benchmark_perf_issues.py``.

``measurement_scope`` separates steady-state operation rows from explicitly
labelled diagnostics (per-call session entry, eager AD workflows, counter
runs); diagnostics never appear in the steady-state tables.

The #1865 and #1863 workloads are einsum instances of ``cpu/einsum``
(``data/instances/bin_omeinsum_*.json``, ``bin_permuted_r4_*.json``) so that
suite's PyTorch/JAX/OMEinsum arms apply; they are generated here too.

The CUDA cases of ``gpu/perf_issues`` (``benchmark_gpu_perf_issues``) are
generated the same way into ``data/instances/gpu_perf_issues.json``,
``benchmarks/gpu/perf_issues.yaml`` and ``benchmarks/gpu/manifests/``.

Run ``python3 scripts/generate_perf_issue_cases.py`` after editing and bump
MANIFEST_VERSION (GPU_MANIFEST_VERSION) whenever a case is added, removed or redefined;
``--check`` fails when the checked-in files are stale.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
INSTANCES = ROOT / "data/instances/perf_issues.json"
MANIFEST = ROOT / "benchmarks/cpu/manifests/perf_issues.yaml"
SUITE = ROOT / "benchmarks/cpu/perf_issues.yaml"
MANIFEST_VERSION = 5
SUITE_ID = "cpu/perf_issues"

GPU_INSTANCES = ROOT / "data/instances/gpu_perf_issues.json"
GPU_MANIFEST = ROOT / "benchmarks/gpu/manifests/perf_issues.yaml"
GPU_SUITE = ROOT / "benchmarks/gpu/perf_issues.yaml"
GPU_MANIFEST_VERSION = 2
GPU_SUITE_ID = "gpu/perf_issues"

DIAGNOSTIC_SCOPES = {"session_entry_diagnostic", "counter_diagnostic", "eager_ad_workflow",
                     "first_call_diagnostic", "state_diagnostic", "issue_workflow_diagnostic"}


def case(cid, issues, kind, params, *, arm, backend="tenferro-rs", dtype, scope="steady_state",
         reference_for=None, profiles=("amd-cpu",), intent):
    row = {
        "id": cid,
        "issues": list(issues),
        "kind": kind,
        "arm": arm,
        "backend": backend,
        "dtype": dtype,
        "params": dict(params, kind=kind, arm=arm, dtype=dtype),
        "measurement_scope": scope,
        "profiles": list(profiles),
        "intent": intent,
    }
    if reference_for:
        row["reference_for"] = reference_for
    return row


def decode_cases():
    issues = ("#1992", "#2003", "#1995")
    intent = ("tenferro-rs #1992 (F1) / #2003 / #1995: decode-shaped projection y = W^T x, "
              "x column-major (in, len), W column-major (in, out), f32. The eager dot_general "
              "wrapper cost and the faer-vs-sgemm provider gap are both visible against the "
              "reference arms in the same family.")
    arms = [
        ("eager-shared", "tenferro-rs", "steady_state", None),
        ("eager-per-call", "tenferro-rs", "session_entry_diagnostic", None),
        ("faer-direct", "faer-direct", "steady_state", "eager-shared"),
        ("host-sgemm", "matrixmultiply-sgemm", "steady_state", "eager-shared"),
        ("pytorch", "pytorch-cpu", "steady_state", "eager-shared"),
    ]
    rows = []
    for out in (1024, 4096):
        for length in (8, 64, 512):
            for arm, backend, scope, ref in arms:
                rows.append(case(
                    f"decode_proj_f32_{arm}_in1024_out{out}_len{length}", issues,
                    "decode_projection", {"in": 1024, "out": out, "len": length},
                    arm=arm, backend=backend, dtype="f32", scope=scope,
                    reference_for=ref and f"decode_proj_f32_{ref}_in1024_out{out}_len{length}",
                    profiles=("amd-cpu", "mac-cpu"), intent=intent))
    return rows


def other_cases():
    rows = [case(
        "decode_block_copy_volume_f32_d1024_len8", ("#1995",), "copy_volume", {"d": 1024, "len": 8},
        arm="eager-counters", dtype="f32", scope="counter_diagnostic",
        intent=("tenferro-rs #1995 ask 2: host allocation count and bytes per eager op of a decode "
                "block (broadcast_in_dim, reshape, transpose, elementwise, reduce, projection), "
                "through public APIs only. Counter run, not timing; per-op layout-copy counts have "
                "no public observation point and are reported as unavailable."))]
    for dtype in ("c64", "f64"):
        rows.append(case(
            f"gemm_{dtype}_mm256_t1", ("#1900",), "gemm_mm256", {"n": 256}, arm="dot-general-shared",
            dtype=dtype, reference_for="gemm_c64_mm256_t1" if dtype == "f64" else None,
            profiles=("mac-cpu", "amd-cpu"),
            intent=("tenferro-rs #1900: 256x256x256 GEMM through CpuBackend::with_threads(1) in one "
                    "session; the c64/f64 time ratio (c64 = 4x the real flops) is the kernel-quality "
                    "signal. The finding is on Apple NEON/FCMA (mac-cpu); the EPYC row is a control.")))
    for dtype in ("f64", "c64"):
        for length in (100, 10_000, 1_000_000):
            for arm, backend, ref in (("dot-general-rank1", "tenferro-rs", None),
                                      ("portable-loop", "rust-loop", "dot-general-rank1"),
                                      ("pytorch-vdot", "pytorch-cpu", "dot-general-rank1")):
                rows.append(case(
                    f"conj_dot_{dtype}_{arm}_len{length}", ("#1615",), "rank1_dot", {"len": length},
                    arm=arm, backend=backend, dtype=dtype,
                    reference_for=ref and f"conj_dot_{dtype}_{ref}_len{length}",
                    intent=("tenferro-rs #1615: conjugated inner product sum(conj(x) y) of two "
                            "contiguous vectors. The rank-1 dot_general route (conjugated operand "
                            "prepared outside timing) is compared with an allocation-free Rust loop "
                            "and torch.vdot (BLAS ddot/zdotc of the PyTorch provider). A direct "
                            "BLAS Level-1 arm from this harness is not linked because the system "
                            "BLAS is only reachable through tenferro's provider; torch.vdot stands in.")))
    for k in (16, 64):
        batch = 64
        for arm, backend, ref in (("solve-loop", "tenferro-rs", None),
                                  ("triangular-solve-loop", "tenferro-rs", None),
                                  ("solve-batched", "tenferro-rs", "solve-loop"),
                                  ("pytorch-solve-batched", "pytorch-cpu", "solve-loop"),
                                  ("pytorch-solve-triangular-batched", "pytorch-cpu",
                                   "triangular-solve-loop")):
            rows.append(case(
                f"small_solve_f64_{arm}_k{k}_b{batch}", ("#2007",), "small_solves",
                {"k": k, "batch": batch}, arm=arm, backend=backend, dtype="f64",
                reference_for=ref and f"small_solve_f64_{ref}_k{k}_b{batch}",
                intent=("tenferro-rs #2007: batch x (k x k) systems solved by a per-item loop of "
                        "rank-2 solve / triangular_solve calls in one session (items pre-sliced "
                        "outside timing). References: the existing batched solve ([k, k, batch]) "
                        "and PyTorch batched solve / solve_triangular. tenferro has no batched "
                        "triangular_solve yet, so that future call has no tenferro row.")))
    for leaves in (0, 2, 32, 128, 512):
        rows.append(case(
            f"eager_backward_matmul2x2_f64_leaves{leaves}", ("#1803",), "backward_live_leaves",
            {"unrelated_leaves": leaves}, arm="eager-ad", dtype="f64", scope="eager_ad_workflow",
            reference_for=None if leaves == 0 else "eager_backward_matmul2x2_f64_leaves0",
            intent=("tenferro-rs #1803 A: forward + sum + backward of one 2x2 matmul with N "
                    "unrelated requires-grad leaves alive in the same eager runtime; N=0 is the "
                    "reference. An eager AD workflow (internal session entries), not a "
                    "steady-state operation row.")))
    for chain in (10, 100, 200):
        for arm, ref in (("compiled-prepared", None), ("eager-shared", "compiled-prepared")):
            rows.append(case(
                f"tanh_chain_f32_{arm}_k{chain}_1024x64", ("#1990",), "tanh_chain",
                {"rows": 1024, "cols": 64, "chain": chain}, arm=arm, dtype="f32",
                reference_for=ref and f"tanh_chain_f32_{ref}_k{chain}_1024x64",
                profiles=("amd-cpu", "mac-cpu"),
                intent=("tenferro-rs #1990: K chained tanh on a 1024x64 f32 tensor, compiled into one "
                        "fused elementwise region (prepared once, run_prepared timed) vs the same "
                        "chain as eager per-op kernels in one session.")))
    for norm in ("layer_norm", "rms_norm"):
        for length in (8, 64):
            for batch in (1, 8):
                for arm, backend, ref in (("eager-composed", "tenferro-rs", None),
                                          ("pytorch-fused", "pytorch-cpu", "eager-composed"),
                                          # #2010 PR-B1: the single-call composite op, a
                                          # second row under test next to eager-composed.
                                          ("eager-single-call", "tenferro-rs", None)):
                    single = arm == "eager-single-call"
                    rows.append(case(
                        f"{norm}_f32_{arm}_d1024_len{length}_b{batch}", ("#2006",), "composed_norm",
                        {"norm": norm, "d": 1024, "len": length, "batch": batch}, arm=arm,
                        backend=backend, dtype="f32",
                        reference_for=ref and f"{norm}_f32_{ref}_d1024_len{length}_b{batch}",
                        intent=(f"tenferro-rs #2006 (#2010 PR-B1): one EagerSession::{norm} call "
                                "(axis 0, weight, bias for layer_norm, eps 1e-5), a composite of "
                                "existing primitives; the eager-composed row is the hand "
                                "composition it replaces (same performance expected until the "
                                "fused B2 kernel)." if single else
                                "tenferro-rs #2006: feature-first (d, len, batch) activation "
                                "normalized over d with weight (and bias for layer_norm). The "
                                "tenferro row is today's ~10-op eager composition and is the "
                                "baseline for the fused B1/B2 op; PyTorch's fused "
                                "layer_norm / rms_norm is the reference.")))
    for arm, scope, ref in (("per-call-default-pool", "session_entry_diagnostic", None),
                            ("per-call-threads1", "session_entry_diagnostic", "per-call-default-pool"),
                            ("shared-session-default-pool", "steady_state", None)):
        rows.append(case(
            f"small_contraction_abcd-dbef-acef_f64_{arm}_d4", ("#1885",),
            "small_contraction_pool_hop", {"extent": 4}, arm=arm, dtype="f64", scope=scope,
            reference_for=ref and f"small_contraction_abcd-dbef-acef_f64_{ref}_d4",
            profiles=("amd-cpu", "mac-cpu"),
            intent=("tenferro-rs #1885 §4.1 addendum: one-shot dot_general abcd,dbef->acef at extent "
                    "4 with a session entered per call on CpuBackend::new() (default pool) vs "
                    "CpuBackend::with_threads(1), and the same call inside one session. Only "
                    "meaningful at more than one thread (with 1 thread new() equals the 1-thread "
                    "backend).")))
    return rows


def b1_op_cases():
    """#2010 PR-B1 composite operations (and the new erf primitive).

    Each op has a single-call arm on the eager surface (``EagerSession``) and on
    the concrete session surface (``TensorSessionOpsExt`` inside one
    ``with_backend_session``), next to a composed reference built from the
    primitives that existed before B1 and, where PyTorch has the op, a
    ``pytorch-cpu`` reference. The ``layer_norm`` / ``rms_norm`` single-call arms
    live in the #2006 ``composed_norm`` family above.
    """
    rows = []
    # erf and exact gelu have no composition from pre-B1 primitives (erf is the
    # new primitive), so they have no eager-composed arm.
    composed = {"sigmoid", "silu", "softplus", "gelu_tanh"}
    rows_, cols = 1024, 64
    for op in ("erf", "sigmoid", "silu", "softplus", "gelu", "gelu_tanh"):
        arms = [("eager-single-call", "tenferro-rs", None),
                ("session-single-call", "tenferro-rs", None)]
        if op in composed:
            arms.append(("eager-composed", "tenferro-rs", "eager-single-call"))
        arms.append(("pytorch", "pytorch-cpu", "eager-single-call"))
        for arm, backend, ref in arms:
            rows.append(case(
                f"activation_{op}_f32_{arm}_{rows_}x{cols}",
                ("#1975", "#2030") if op == "gelu" else ("#1975",), "activation",
                {"op": op, "rows": rows_, "cols": cols}, arm=arm, backend=backend, dtype="f32",
                reference_for=ref and f"activation_{op}_f32_{ref}_{rows_}x{cols}",
                intent=(f"tenferro-rs #1975 (#2010 PR-B1): elementwise {op} on a {rows_}x{cols} "
                        "f32 tensor as one eager call and as one concrete session call "
                        "(TensorSessionOpsExt). "
                        + ("erf is the new B1 primitive (libm erff on CPU). " if op == "erf" else "")
                        + ("The eager-composed arm is the hand-written formula from existing "
                           "primitives (exp, neg, abs, log1p, tanh, maximum, ...). " if op in composed else
                           "No pre-B1 composition exists (it needs erf), so the reference is "
                           "PyTorch only. ")
                        + "PyTorch is the external reference."
                        + (" #2030 tracks the confirmed borrowed-session exact GELU CPU gap." if op == "gelu" else ""))))
    for row in rows:
        if row["params"]["op"] == "gelu":
            continue  # Exact GELU belongs to the existing #2030 report.
        row["issues"].append("#2032")
        row["intent"] += " #2032 tracks this confirmed current-main activation workload."
        if row["arm"] in ("eager-single-call", "pytorch"):
            row["followup_mwe"] = dict(case_id="activation." + row["params"]["op"],
                                      rust_path="eager-shared", reference_path="pytorch",
                                      path="eager-shared" if row["arm"] == "eager-single-call" else "pytorch")
            row["execution_path"] = row["followup_mwe"]["path"]
    for op in ("softmax", "log_softmax", "masked_softmax"):
        for length in (64, 512):
            batch = 8
            for arm, backend, ref in (("eager-single-call", "tenferro-rs", None),
                                      ("session-single-call", "tenferro-rs", None),
                                      ("eager-composed", "tenferro-rs", "eager-single-call"),
                                      ("pytorch", "pytorch-cpu", "eager-single-call")):
                rows.append(case(
                    f"{op}_f32_{arm}_len{length}_b{batch}",
                    ("#1976", "#2031") if op == "softmax" and length == 64 else ("#1976",), "softmax",
                    {"op": op, "len": length, "batch": batch, "axis": 0}, arm=arm,
                    backend=backend, dtype="f32",
                    reference_for=ref and f"{op}_f32_{ref}_len{length}_b{batch}",
                    intent=(f"tenferro-rs #1976 (#2010 PR-B1): {op} over axis 0 of an "
                            f"attention-score-like (key {length}, query {length}, batch {batch}) "
                            "f32 tensor"
                            + (" with a causal Bool mask (key <= query) broadcast over the batch "
                               "as (len, len, 1)" if op.startswith("masked") else "")
                            + ", as one eager call and one concrete session call. The "
                            "eager-composed arm is the plain max-subtracted composition from "
                            "existing primitives (reduce_max, broadcast_in_dim, sub, exp, "
                            "reduce_sum, div/log"
                            + (", select for the mask, which is pre-broadcast to the full "
                               "(len, len, batch) shape outside timing" if op.startswith("masked")
                               else "")
                            + ") without the all-masked guard; PyTorch is the external "
                            "reference."
                            + (" #2031 tracks the confirmed borrowed-session len64 CPU softmax gap."
                               if op == "softmax" and length == 64 else ""))))
    for arm, backend, ref in (("eager-single-call", "tenferro-rs", None),
                              ("session-single-call", "tenferro-rs", None),
                              ("eager-composed", "tenferro-rs", "eager-single-call"),
                              ("pytorch", "pytorch-cpu", "eager-single-call")):
        rows.append(case(
            f"reduce_mean_f32_{arm}_1024x64_axis0", ("#1976",), "reduce_mean",
            {"rows": 1024, "cols": 64, "axis": 0}, arm=arm, backend=backend, dtype="f32",
            reference_for=ref and f"reduce_mean_f32_{ref}_1024x64_axis0",
            intent=("tenferro-rs #1976 (#2010 PR-B1): reduce_mean over axis 0 of a 1024x64 f32 "
                    "tensor as one eager call and one concrete session call; the composed "
                    "reference is reduce_sum + scale_real, PyTorch mean is the external "
                    "reference.")))
    n, batch = 64, 256
    for arm, backend, ref in (("eager-single-call", "tenferro-rs", None),
                              ("session-single-call", "tenferro-rs", None),
                              ("eager-gather-prebuilt", "tenferro-rs", "eager-single-call"),
                              ("pytorch", "pytorch-cpu", "eager-single-call")):
        rows.append(case(
            f"take_along_axis_rows_f64_{arm}_n{n}_b{batch}", ("#2008",), "take_along_axis",
            {"n": n, "batch": batch, "axis": 0}, arm=arm, backend=backend, dtype="f64",
            reference_for=ref and f"take_along_axis_rows_f64_{ref}_n{n}_b{batch}",
            intent=(f"tenferro-rs #2008 (#2010 PR-B1): per-batch row gather "
                    f"out[i, j, b] = x[idx[i, b], j, b] on x [{n}, {n}, {batch}] f64 with an "
                    f"I64 index [{n}, 1, {batch}] (a different row permutation per batch), "
                    "as one take_along_axis call (eager and concrete session; the composite "
                    "builds the (index, batch) tuples inside the call). The "
                    "eager-gather-prebuilt arm calls the existing gather with the tuples built "
                    "outside timing, isolating the index-construction cost; PyTorch "
                    "take_along_dim is the external reference.")))
    return rows


def generate_cases():
    return decode_cases() + other_cases() + b1_op_cases() + cpu_gap_cases() + cpu_followup_cases()


def cpu_gap_cases():
    """Public borrowed-session MWE and matched PyTorch references."""
    rows = []
    for operation, issue, suffix, arm, intent in (
        ("lstsq", "#2027", "m768_n384_rhs16", "eager-shared",
         "Full-column-rank f64 least squares; analytical solution is all ones. "
         "EagerSession::lstsq borrows one session outside timing; reference is "
         "PyTorch unpivoted gels, retaining the full public result."),
        ("triangular", "#2028", "n4096_rhs64", "concrete-shared",
         "Single rank-2 lower-triangular f64 solve, 64 RHS columns, no AD. "
         "One CPU execution session surrounds every sample; analytical solution is all ones. "
         "Reference: torch.linalg.solve_triangular(upper=False)."),
        ("reshape", "#2029", "n33554432_to8192x4096", "concrete-shared",
         "Materializing reshape_read of a borrowed owned f64 tensor returns a fresh "
         "column-major [8192,4096] output. Reference: clone of the matching PyTorch "
         "view, prepared outside timing; metadata-only reshape is not the comparator."),
    ):
        target = f"{operation}_f64_{arm}_{suffix}"
        for name, backend in ((arm, "tenferro-rs"), ("pytorch", "pytorch-cpu")):
            rows.append(case(f"{operation}_f64_{name}_{suffix}", (issue,), "cpu_gap_mwe",
                {"operation": operation}, arm=name, backend=backend, dtype="f64",
                reference_for=target if backend == "pytorch-cpu" else None,
                profiles=("amd-cpu", "mac-cpu"), intent=f"tenferro-rs {issue}: {intent}"))
    return rows



def cpu_followup_cases():
    """Confirmed current-main public operations and their independent references.

    The standalone crate avoids the legacy root runner's provider-feature paths.
    Exact activation workloads already exist in b1_op_cases; annotate them there.
    """
    catalog = {c["id"]: c for c in json.loads(
        (ROOT / "mwe/cpu_followup/cases.json").read_text())}
    groups = {2021: ['structural.broadcast_in_dim'],
     2033: ['real.tanh',
            'real.log1p',
            'real.cos',
            'real.sin',
            'real.exp',
            'real.expm1',
            'real.pow',
            'complex.exp',
            'real.log'],
     2034: ['real.reduce_max_axis1',
            'real.reduce_prod_axis1',
            'real.reduce_sum_axis1',
            'real.reduce_max_axis0',
            'real.reduce_min_axis1',
            'real.reduce_min_axis0',
            'real.reduce_max_all',
            'real.reduce_min_all'],
     2035: ['complex.norm_fro'],
     2036: ['index.gather', 'index.scatter', 'index.dynamic_update_slice'],
     2037: ['perm.reverse_15d_3',
            'structural.transpose',
            'perm.reverse_23d_2',
            'perm.transpose_3d_256_201',
            'perm.cyclic_15d_3',
            'perm.transpose_3d_256_102',
            'perm.tn_light_415_24d_scattered_to_colmajor',
            'perm.tn_light_415_24d_contiguous_same_perm',
            'perm.rotation_6d_32_32_32_32_16_16'],
     2038: ['structural.extract_diagonal'],
     2039: ['fft.irfft', 'fft.rfft', 'fft.fft', 'fft.ifft'],
     2040: ['metadata.transpose_view', 'metadata.slice_view', 'metadata.reshape_view']}
    rows = []
    for issue, ids in groups.items():
        for cid in ids:
            c = catalog[cid]
            path = c["paths"][0]
            if c["category"] == "fft":
                reference = "mkl-dfti"
            elif c["category"] == "perm":
                reference = "strided-rs"
            elif c["category"] == "metadata" or c["source_reference"] == "julia-base":
                reference = "julia-base"
            elif cid == "index.dynamic_update_slice" or c["source_reference"] == "jax-cpu":
                reference = "jax"
            else:
                reference = "pytorch"
            params = dict(case_id=cid, rust_path=path, reference_path=reference,
                          fixture=c["fixture"])
            target = "followup_" + cid.replace(".", "_") + "_" + path
            for arm in (path, reference):
                backend = "tenferro-rs" if arm == path else {
                    "pytorch": "pytorch-cpu", "jax": "jax-cpu",
                    "julia-base": "julia-base", "strided-rs": "strided-rs",
                    "mkl-dfti": "mkl-dfti"}[arm]
                row = case("followup_" + cid.replace(".", "_") + "_" + arm,
                           (f"#{issue}",), "cpu_followup_mwe", params,
                           arm=arm, backend=backend, dtype=c["dtype"],
                           reference_for=target if arm != path else None,
                           intent=f"tenferro-rs #{issue}: {cid}, fixed public-API fixture "
                                  f"{c['fixture']}; setup untimed, outputs retained, "
                                  "borrowed session except pure metadata. Paired reference "
                                  "path is separately labelled; trace is scope-ineligible.")
                row["execution_path"] = ("jax-compiled" if arm == "jax" else
                                         "mkl-dfti-cached" if arm == "mkl-dfti" else arm)
                rows.append(row)
            if issue == 2040:
                original = rows[-2]
                for typed_path in c["paths"][1:]:
                    import copy
                    typed = copy.deepcopy(original)
                    typed["id"] = "followup_" + cid.replace(".", "_") + "_" + typed_path
                    typed["params"]["rust_path"] = typed_path
                    typed["params"]["arm"] = typed_path
                    typed["arm"] = typed_path
                    typed["execution_path"] = typed_path
                    typed["intent"] = (f"tenferro-rs #2040: {cid}; borrowed TypedTensorView<f64, R, Host>; "
                        "separate owners and all outputs retained; as_view untimed; same Julia reference fixture. "
                        "Static reshape input Rank<1> returns DynRank; slice/transpose retain rank.")
                    rows.append(typed)
    return rows

def generate_gpu_cases():
    """gpu/perf_issues (CUDA): run by benchmark_gpu_perf_issues on the local device."""
    rows = []
    gpu = ("nvidia-gpu",)
    for direction in ("up", "down"):
        for size in (8, 4 << 10, 64 << 10, 1 << 20, 16 << 20, 256 << 20):
            for arm, backend, ref in (("tenferro", "tenferro-cuda", None),
                                      ("cudarc-pageable", "cudarc", "tenferro"),
                                      ("cudarc-pinned", "cudarc", "tenferro")):
                rows.append(case(
                    f"transfer_{direction}_f64_{arm}_{size}B", ("#2009",), "transfer",
                    {"bytes": size, "direction": direction}, arm=arm, backend=backend, dtype="f64",
                    reference_for=ref and f"transfer_{direction}_f64_{ref}_{size}B", profiles=gpu,
                    intent=("tenferro-rs #2009: host-device transfer of one f64 buffer (8 B to "
                            "256 MiB), upload_tensor + runtime synchronize / download_tensor, vs "
                            "plain cudarc memcpy from pageable and pinned host memory on the same "
                            "device. Every call includes its own synchronize; the transfer is the "
                            "declared operation.")))
    for elements in (0, 16, 512, 32768, 2 << 20):
        for arm, ref in (("alloc-zero-within-callback", None), ("alloc-zero-per-callback", None),
                         ("host-zero-upload", "alloc-zero-per-callback"),
                         ("alloc-fill-per-callback", "alloc-zero-per-callback")):
            rows.append(case(
                f"alloc_zero_f64_{arm}_{elements}el", ("#1887",), "alloc_zero", {"elements": elements},
                arm=arm, backend="tenferro-cuda", dtype="f64",
                reference_for=ref and f"alloc_zero_f64_{ref}_{elements}el", profiles=gpu,
                intent=("tenferro-rs #1887: Session::alloc_zero_output (0, 16 elements, 4 KiB, "
                        "256 KiB, 16 MiB) many times inside one with_cubecl callback and once per "
                        "callback, vs uploading a host zero buffer and vs alloc_output + "
                        "fill_zero_write (device fill) per callback. Each batch ends with one device "
                        "synchronize.")))
    for n in (8, 16, 32, 64):
        for arm, ref in (("loop", None), ("batched", "loop")):
            rows.append(case(
                f"small_blocks_gemm_f64_{arm}_n{n}_count256", ("#1885",), "small_blocks",
                {"n": n, "count": 256}, arm=arm, backend="tenferro-cuda", dtype="f64",
                reference_for=ref and f"small_blocks_gemm_f64_{ref}_n{n}_count256", profiles=gpu,
                intent=("tenferro-rs #1885 §3.1/§3.2: 256 independent n x n GEMMs submitted one "
                        "dot_general per block vs one batched dot_general call (the reference); "
                        "the submission cost of many small blocks.")))
    for op in ("qr", "svd"):
        for batch in (16, 256):
            for arm, ref in (("loop", None), ("batched", "loop")):
                rows.append(case(
                    f"batched_{op}_f64_{arm}_n32_b{batch}", ("#1885",), "batched_factorization",
                    {"op": op, "n": 32, "batch": batch}, arm=arm, backend="tenferro-cuda", dtype="f64",
                    reference_for=ref and f"batched_{op}_f64_{ref}_n32_b{batch}", profiles=gpu,
                    intent=("tenferro-rs #1885 §3.3: batch x (32 x 32) QR / SVD as a per-matrix "
                            "loop vs one call on a [n, n, batch] tensor (whose CUDA implementation "
                            "may itself loop with per-block info syncs). Time only; the sync count "
                            "per block has no public counter and is reported as unavailable.")))
    for live in (16, 256, 1024):
        rows.append(case(
            f"qr_live_buffers_f64_n32_live{live}", ("#1885",), "batched_factorization",
            {"op": "qr", "n": 32, "batch": 1, "live_buffers": live}, arm="loop",
            backend="tenferro-cuda", dtype="f64",
            reference_for=None if live == 16 else "qr_live_buffers_f64_n32_live16", profiles=gpu,
            intent=("tenferro-rs #1885 §3.3 addendum: host cost of one 32 x 32 QR while 16 / 256 / "
                    "1024 other device buffers are resident; the 16-buffer case is the reference.")))
    for arm in ("to-contiguous", "copy-into"):
        rows.append(case(
            f"first_call_transpose_{arm}_f64_layouts_r2-12_c5-7", ("#1885",), "first_call_layouts",
            {"max_rows": 12, "cols": [5, 7]}, arm=arm, backend="tenferro-cuda", dtype="f64",
            scope="first_call_diagnostic", profiles=gpu,
            intent=("tenferro-rs #1885 §3.4 addendum: first and second call of a materializing "
                    "transpose (TensorViewCanonicalization::to_contiguous, or copy_into a compact "
                    "destination, of a transposed device view) for 22 distinct [r, c] layouts after "
                    "one warm-up layout, in one cold process. A first-call (NVRTC compile) "
                    "diagnostic, never a steady-state row; the second call is the warm reference.")))
    rows += gpu_latency_cases(gpu)
    rows += gpu_memory_cases(gpu)
    rows += gpu_linalg_driver_cases(gpu)
    rows += gpu_contraction_cases(gpu)
    return rows


# ---------------------------------------------------------------------------
# #2078 / #2079: raw-session entry and eager per-op host cost
# ---------------------------------------------------------------------------

def gpu_latency_cases(gpu):
    """Workloads of #2078 (raw-session entry) and #2079 (decode-sized eager ops)."""
    rows = []
    for arm, ref in (("empty-idle", None),
                     ("empty-behind-queued-gemm", "sync-behind-queued-gemm"),
                     ("sync-behind-queued-gemm", None)):
        rows.append(case(
            f"raw_scope_entry_f64_{arm}", ("#2078",), "raw_scope_entry",
            {"gemm_n": 4096, "queued_gemms": 4}, arm=arm,
            backend="tenferro-cuda", dtype="f64", scope="session_entry_diagnostic",
            reference_for=ref and f"raw_scope_entry_f64_{ref}", profiles=gpu,
            intent=("tenferro-rs #2078: entering a raw CUDA session "
                    "(CudaExecSession::with_raw) with an empty closure, on an idle device and "
                    "behind queued device work (4 x 4096^3 f64 GEMMs enqueued first). The issue "
                    "measures the second empty scope after the GEMMs, which blocks for the whole "
                    "device time; the sync arm is the explicit barrier it is compared with. The idle "
                    "arm runs many scopes in one timed interval; the queued arms hold one scope per "
                    "batch (the retention precondition caps the batch), so their per-op number is that "
                    "call. A raw scope must return in microseconds and leave queued work in flight.")))
    for op, shape in (("add", {"hidden": 1024, "tokens": 8}),
                      ("dot_general", {"in": 1024, "out": 1024, "len": 8})):
        label = f"{shape.get('hidden', shape.get('in'))}x{shape.get('tokens', shape.get('len'))}"
        for arm, scope, ref in (("eager-session", "steady_state", None),
                                ("eager-per-call", "session_entry_diagnostic", "eager-session")):
            rows.append(case(
                f"eager_op_f32_{op}_decode_{arm}", ("#2079",), "eager_op",
                dict(shape, op=op), arm=arm, backend="tenferro-cuda", dtype="f32", scope=scope,
                reference_for=ref and f"eager_op_f32_{op}_decode_{ref}", profiles=gpu,
                intent=(f"tenferro-rs #2079 ({label} f32, decode-sized): the host-side cost of one "
                        "eager op on CUDA. `eager-session` times the op in one reused "
                        "with_eager_session with a single device synchronize at the batch edge, "
                        "so the per-op nanoseconds are the host dispatch the issue measured "
                        "(11.5 us per add, 41.3 us per GEMM) plus the amortized wait; "
                        "`eager-per-call` enters a session per operation, the diagnostic that "
                        "separates session entry from dispatch. The GPU work of one decode-shaped "
                        "op is a small fraction of the host cost, so these are latency rows, not "
                        "throughput.")))
    return rows


# ---------------------------------------------------------------------------
# #2080 / #2081 / #2082 / #2083 / #1924: table-sized materialization,
# device memory, per-session vendor handles and vendor churn
# ---------------------------------------------------------------------------

def gpu_memory_cases(gpu):
    """Workloads of #2080 (gather), #2081 (freed memory), #2082 (reshape), #2083, #1924."""
    rows = []
    for vocab in (8192, 65536, 248320):
        for arm, ref, backend in (("eager-gather", None, "tenferro-cuda"),
                                  ("cudarc-row-copies", "eager-gather", "cudarc")):
            rows.append(case(
                f"gather_embedding_f32_vocab{vocab}_{arm}", ("#2080",), "gather_table",
                {"vocab": vocab, "hidden": 1024, "tokens": 8}, arm=arm,
                backend=backend, dtype="f32",
                reference_for=ref and f"gather_embedding_f32_vocab{vocab}_{ref}", profiles=gpu,
                intent=("tenferro-rs #2080: an embedding lookup (8 token ids, hidden=1024, f32) "
                        "from a [vocab, hidden] table, vs the 8 device-to-device row copies the "
                        "same selection needs, into a freshly allocated destination so the two "
                        "arms have the same output policy. The issue measures 0.23 / 1.67 / 6.3 ms at vocab "
                        "8192 / 65536 / 248320, i.e. time proportional to the table because the "
                        "whole table is permuted first; the row-copy arm shows the O(tokens) cost "
                        "the lookup is expected to have.")))
    for arm, ref in (("constant", None), ("op-output", "constant")):
        rows.append(case(
            f"reshape_constant_f32_64M_{arm}", ("#2082",), "reshape_materialize",
            {"elements": 1 << 26, "rows": 1 << 16, "cols": 1 << 10}, arm=arm,
            backend="tenferro-cuda", dtype="f32",
            reference_for=ref and f"reshape_constant_f32_64M_{ref}", profiles=gpu,
            intent=("tenferro-rs #2082: reshape of a 256 MiB f32 device tensor from [2^26] to "
                    "[2^16, 2^10]. The issue measured the reshape of an uploaded constant as a "
                    "cuTENSOR permute copy plus a new 256 MiB buffer (0.5-0.65 ms) while the "
                    "reshape of an op output (e.g. add) was a metadata-only view (0.04 ms); "
                    "`constant` reshapes the uploaded constant and `op-output` reshapes an "
                    "elementwise result, with the op-output arm as the reference the constant "
                    "route has to reach. A compact reshape must be a view, not a device copy.")))
    for arm, ref in (("create-destroy", None), ("reused", "create-destroy")):
        rows.append(case(
            f"vendor_handle_cublas_f64_{arm}", ("#2083",), "vendor_handle",
            {"arm": arm}, arm=arm, backend="tenferro-cuda", dtype="f64",
            scope="session_entry_diagnostic",
            reference_for=ref and f"vendor_handle_cublas_f64_{ref}", profiles=gpu,
            intent=("tenferro-rs #2083: the documented raw-session pattern creates and destroys "
                    "a cuBLAS handle in every session (cublasCreate + cublasDestroy, 0.45 ms and "
                    "10 MiB per session on the issue's card); the reused arm keeps one handle. "
                    "Both arms are per-iteration inside the raw scope.")))
    for cycles in (8, 64):
        rows.append(case(
            f"vendor_churn_f64_cycles{cycles}", ("#1924",), "vendor_churn",
            {"cycles": cycles, "gemm_n": 256}, arm="backend-cycle", backend="tenferro-cuda",
            dtype="f64", scope="state_diagnostic", profiles=gpu,
            intent=("tenferro-rs #1924: creating and dropping a CudaBackend (which dlopens and "
                    "unloads the vendor libraries) N times, reporting the driver free memory "
                    "before and after. The issue measured ~55 MB leaked per cycle; a state "
                    "diagnostic, not a timing row.")))
    for arm in ("alloc-free", "runtime-recreate", "driver-alloc-after-free"):
        rows.append(case(
            f"pool_retention_f64_{arm}", ("#2081",), "pool_retention",
            {"arm": arm, "gib": 1, "count": 8, "probe_gib": 6}, arm=arm,
            backend="tenferro-cuda", dtype="f64", scope="state_diagnostic", profiles=gpu,
            intent=("tenferro-rs #2081: device memory of dropped CUDA tensors (and of a dropped "
                    "runtime) is never returned to the driver. The issue's sequence: eight 1 GiB "
                    "tensors alive at the high-water mark (one of them downloaded and compared "
                    "with its host payload), then all dropped. The driver free memory is reported "
                    "before, at the high-water mark, after the drops, and after recreating the "
                    "runtime; `driver-alloc-after-free` adds the synchronous 6 GiB driver "
                    "allocation the issue could not serve. State diagnostic: bytes, not "
                    "nanoseconds. A caller must see the memory back or get an Err, never a panic "
                    "with Ok from alloc_output.")))
    return rows


# ---------------------------------------------------------------------------
# #1839 / #1846 / #1849 / #1852 / #1923: CUDA linalg driver and batch knees
# ---------------------------------------------------------------------------

def gpu_linalg_driver_cases(gpu):
    """Workloads of the cuSOLVER driver-selection issues."""
    rows = []
    for n in (64, 256, 1024):
        for driver in ("auto", "gesvdj", "gesvd", "xgesvdp"):
            rows.append(case(
                f"svd_driver_c64_n{n}_{driver}", ("#1839", "#1849"), "svd_driver",
                {"n": n, "driver": driver}, arm=driver, backend="tenferro-cuda", dtype="c64",
                reference_for=None if driver == "auto" else f"svd_driver_c64_n{n}_auto",
                profiles=gpu,
                intent=("tenferro-rs #1839 / #1849: full SVD of a complex f64 matrix with an "
                        "explicit cuSOLVER driver on a DMRG-like c64 input. Auto is the "
                        "JAX-compatible policy (gesvdj at both dimensions <= 1024), which the "
                        "issue measured as 4-5x slower than gesvd below 1024; xgesvdp is the "
                        "polar-decomposition driver added in #1849 and is explicit-only.")))
    for dtype in ("f64", "c64"):
        for n in (16, 32, 128):
            for driver in ("auto", "syevd", "syevj"):
                rows.append(case(
                    f"eigh_batched_{dtype}_n{n}_{driver}", ("#1846", "#1852", "#1923"),
                    "eigh_driver", {"n": n, "batch": 64, "driver": driver}, arm=driver,
                    backend="tenferro-cuda", dtype=dtype,
                    reference_for=None if driver == "auto" else f"eigh_batched_{dtype}_n{n}_auto",
                    profiles=gpu,
                    intent=("tenferro-rs #1846 / #1852 / #1923: batched Hermitian "
                            "eigendecomposition of 64 n x n items with an explicit cuSOLVER "
                            "driver. Auto selects cusolverDnXsyevBatched for a batch (about 100x "
                            "to 280x faster than the previous per-matrix loop on the issue's "
                            "card); #1852 records the syevjBatched n <= 32 knee, so n=32 is the "
                            "row that matters and n=16 / n=128 bracket it. #1923 is the c64 "
                            "coverage.")))
    return rows


# ---------------------------------------------------------------------------
# #1898 / #1873 / #1805 / #1949: contraction shape and plan/route lifetime
# ---------------------------------------------------------------------------

def gpu_contraction_cases(gpu):
    """Workloads of the degenerate-contraction and route-lifetime issues."""
    rows = []
    for arm, ref in (("dot-general", None), ("multiply", "dot-general")):
        rows.append(case(
            f"hadamard_f32_1024x1024_{arm}", ("#1898",), "hadamard_dot_general",
            {"rows": 1024, "cols": 1024}, arm=arm, backend="tenferro-cuda", dtype="f32",
            reference_for=ref and f"hadamard_f32_1024x1024_{ref}", profiles=gpu,
            intent=("tenferro-rs #1898: a pure Hadamard product routed through dot_general "
                    "(every axis is a batch axis) vs the elementwise multiply. On the CPU backend "
                    "the former degenerated to one 1x1x1 GEMM per element (23x); the CUDA arm checks "
                    "whether the same shape routing holds on the device.")))
    for arm, ref in (("distinct-plans", None), ("repeated-plan", "distinct-plans")):
        rows.append(case(
            f"contraction_plan_churn_f64_{arm}", ("#1873", "#1805"), "contraction_plan_churn",
            {"distinct": 72, "n": 64}, arm=arm, backend="tenferro-cuda", dtype="f64",
            scope="issue_workflow_diagnostic",
            reference_for=ref and f"contraction_plan_churn_f64_{ref}", profiles=gpu,
            intent=("tenferro-rs #1873 / #1805: a cuTENSOR per-plan workspace can evict the "
                    "contraction-plan cache, and raw-session resources are dropped when a plan is "
                    "evicted. `distinct-plans` fills the cache with 72 distinct small shapes and "
                    "then re-runs the first (more than the backend's 64-plan capacity, so the "
                    "churn really evicts); `repeated-plan` runs the same number of contractions of "
                    "one shape. The timed operation is the re-run alone, with the churn as its "
                    "untimed precondition. A re-run after churn must not cost more than the "
                    "repeated shape. A cache miss reconstructs the contraction plan and its "
                    "workspace inside the call, and the precondition cannot be batched, so this "
                    "is one isolated call per sample: a cache-miss diagnostic, not a steady-state "
                    "operation row.")))
    for arm, ref in (("owned-dst", None), ("borrowed-dst", "owned-dst")):
        rows.append(case(
            f"dot_general_destination_f64_{arm}", ("#1949",), "owned_destination",
            {"n": 512}, arm=arm, backend="tenferro-cuda", dtype="f64",
            scope="issue_workflow_diagnostic",
            reference_for=ref and f"dot_general_destination_f64_{ref}", profiles=gpu,
            intent=("tenferro-rs #1949: the owned-destination memo fast path for raw CUDA vendor "
                    "calls, where a vendor write to a memoized address can overtake a queued "
                    "CubeCL read (the skipped get_resource round trip is what flushes). "
                    "Both arms write one caller-provided destination that is reused by every "
                    "call: `owned-dst` through the owned-tensor write route the memo covers, "
                    "`borrowed-dst` through the borrowed-view route, which keeps the resource "
                    "round trip. The fix must not add a blocking round trip, so the owned arm "
                    "must stay at or below the borrowed one. The destination handle is built per "
                    "call and the precondition cannot be batched, so this is one isolated call "
                    "per sample: a route-lifetime diagnostic, not a steady-state operation row, "
                    "and it reports the destination contents it actually wrote.")))
    return rows


# ---------------------------------------------------------------------------
# cpu/einsum instances (#1865, #1863)
# ---------------------------------------------------------------------------

LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def omeinsum_binary(rank_a, rank_b, contracted, batch):
    """Index layout of omeinsum-rs benches/support/binary.rs (all extents 2)."""
    left = rank_a - contracted - batch
    right = rank_b - contracted - batch
    it = iter(LETTERS)
    lft = [next(it) for _ in range(left)]
    con = [next(it) for _ in range(contracted)]
    rgt = [next(it) for _ in range(right)]
    bat = [next(it) for _ in range(batch)]
    a, b, c = lft + con + bat, con + rgt + bat, lft + rgt + bat
    return "".join(a) + "," + "".join(b) + "->" + "".join(c), [[2] * len(a), [2] * len(b)]


def einsum_instance(name, fmt, shapes, intent):
    inputs, output = fmt.split("->")
    labels = inputs.split(",")
    extents = {}
    for operand, shape in zip(labels, shapes):
        for label, extent in zip(operand, shape):
            extents[label] = extent
    size = math.prod(extents[ch] for ch in output) if output else 1
    flops = math.prod(extents.values())
    path = {"path": [[0, 1]], "log2_size": round(math.log2(size), 4),
            "log10_flops": round(math.log10(2 * flops), 4)}
    return {
        "name": name,
        "format_string": fmt,
        "shapes": shapes,
        "dtype": "float64",
        "num_tensors": 2,
        "paths": {"opt_size": path, "opt_flops": dict(path)},
        "format_string_colmajor": ",".join(op[::-1] for op in labels) + "->" + output[::-1],
        "shapes_colmajor": [list(reversed(s)) for s in shapes],
        "format_string_rowmajor": fmt,
        "intent": intent,
    }


def einsum_instances():
    out = {}
    for name, ra, rb, con, bat in (("matmul_10x10", 10, 10, 5, 0),
                                   ("batched_matmul_8x8_batch_4", 8, 8, 4, 4),
                                   ("high_d_12x12_contract_4_batch_4", 12, 12, 4, 4)):
        fmt, shapes = omeinsum_binary(ra, rb, con, bat)
        out[f"bin_omeinsum_{name}"] = einsum_instance(
            f"bin_omeinsum_{name}", fmt, shapes,
            f"tenferro-rs #1865: omeinsum-rs binary scenario {name} (rank {ra} x rank {rb}, "
            f"{con} contracted and {bat} batch modes, every extent 2), originally f32; this suite "
            "is f64. Small, batched, materialization-heavy contraction; PyTorch is the reference.")
    out["bin_permuted_r4_abcd_dbef_acef_d32"] = einsum_instance(
        "bin_permuted_r4_abcd_dbef_acef_d32", "abcd,dbef->acef", [[32] * 4, [32] * 4],
        "tenferro-rs #1863: representative standard contraction (permuted rank-4 abcd,dbef->acef "
        "at extent 32, 2 GFLOP) for the t1 vs t4/t8 thread-scaling question; the reference is the "
        "same row at 1 thread (run_all.sh 1 / 4, optionally 8).")
    return out


def render(cases, *, suite_id=SUITE_ID, manifest_version=MANIFEST_VERSION,
           source="data/instances/perf_issues.json", device="cpu"):
    full = [c["id"] for c in cases]
    quick = [c["id"] for c in cases if c["measurement_scope"] not in DIAGNOSTIC_SCOPES]
    header = "# Generated by scripts/generate_perf_issue_cases.py; do not edit.\n"
    instances = json.dumps(cases, indent=1, ensure_ascii=False) + "\n"
    manifest = header + (
        "# quick: steady-state operation rows and their reference arms; full adds the\n"
        "# explicitly labelled diagnostics (per-call session entry, eager AD workflows,\n"
        "# counter runs). Bump MANIFEST_VERSION in the generator on any change.\n"
    ) + yaml.safe_dump({
        "manifest_version": manifest_version, "suite_id": suite_id,
        "source": source,
        "generated_by": "scripts/generate_perf_issue_cases.py",
        "coverage": {"quick": quick, "full": full},
    }, sort_keys=False, width=100)
    issues = sorted({i for c in cases for i in c["issues"]}, key=lambda s: int(s[1:]))
    suite = header + yaml.safe_dump({
        "schema_version": 1,
        "suite_id": suite_id,
        "title": f"Workloads of open tenferro-rs performance issues ({device.upper()})",
        "description": (
            "One case family per open tenferro-rs performance issue ("
            + ", ".join(issues)
            + "), each with its reference arm(s); the issue numbers are recorded per case in "
            + source + ". Workloads are added when an issue is opened; "
            "perf-fix PRs report numbers from these cases. Steady-state rows and labelled "
            "diagnostics are reported separately."
            + (" #1865 and #1863 live in cpu/einsum." if device == "cpu" else
               " Timed batches end with a device synchronize, never a result download, except "
               "where the download is the declared operation (#2009).")),
        "defaults": {
            "device": {"kind": "cpu" if device == "cpu" else "cuda", "ordinal": 0},
            "run": {"warmups": 3, "runs": 15 if device == "cpu" else 7, "min_runtime_ms": 2,
                    "timing_scope": "steady_state_host_api" if device == "cpu"
                    else "steady_state_host_api_plus_device_sync"},
            "effort": {"scan": {"warmups": 1, "runs": 3}},
            "verify": {"reference": "independent f64 scalar references, residuals and analytic gradients",
                       "rtol": 1.0e-4, "atol": 0},
        },
        "backends": sorted({c["backend"] for c in cases}),
        "problems": {"source": source, "include": full},
    }, sort_keys=False, width=100)
    return instances, manifest, suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if checked-in files are stale")
    args = parser.parse_args()
    instances, manifest, suite = render(generate_cases())
    files = [(INSTANCES, instances), (MANIFEST, manifest), (SUITE, suite)]
    gpu = render(generate_gpu_cases(), suite_id=GPU_SUITE_ID, manifest_version=GPU_MANIFEST_VERSION,
                 source="data/instances/gpu_perf_issues.json", device="gpu")
    files += list(zip((GPU_INSTANCES, GPU_MANIFEST, GPU_SUITE), gpu))
    for name, inst in einsum_instances().items():
        files.append((ROOT / f"data/instances/{name}.json", json.dumps(inst, indent=2) + "\n"))
    if args.check:
        stale = [p for p, text in files if not p.exists() or p.read_text() != text]
        for path in stale:
            print(f"stale: {path.relative_to(ROOT)}", file=sys.stderr)
        return 1 if stale else 0
    for path, text in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
