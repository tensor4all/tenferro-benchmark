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
MANIFEST_VERSION = 1
SUITE_ID = "cpu/perf_issues"

GPU_INSTANCES = ROOT / "data/instances/gpu_perf_issues.json"
GPU_MANIFEST = ROOT / "benchmarks/gpu/manifests/perf_issues.yaml"
GPU_SUITE = ROOT / "benchmarks/gpu/perf_issues.yaml"
GPU_MANIFEST_VERSION = 1
GPU_SUITE_ID = "gpu/perf_issues"

DIAGNOSTIC_SCOPES = {"session_entry_diagnostic", "counter_diagnostic", "eager_ad_workflow",
                     "first_call_diagnostic"}


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
                                          ("pytorch-fused", "pytorch-cpu", "eager-composed")):
                    rows.append(case(
                        f"{norm}_f32_{arm}_d1024_len{length}_b{batch}", ("#2006",), "composed_norm",
                        {"norm": norm, "d": 1024, "len": length, "batch": batch}, arm=arm,
                        backend=backend, dtype="f32",
                        reference_for=ref and f"{norm}_f32_{ref}_d1024_len{length}_b{batch}",
                        intent=("tenferro-rs #2006: feature-first (d, len, batch) activation "
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


def generate_cases():
    return decode_cases() + other_cases()


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
