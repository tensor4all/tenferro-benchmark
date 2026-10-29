#!/usr/bin/env python3
"""Run the cpu/perf_issues cases (tenferro-rs performance-issue workloads).

Rust cases run one process each through ``perf_issue_case``; ``pytorch-cpu``
reference cases run in this process. Both use the same sampling scheme:
warmups, then calibration toward ``min_runtime_ms`` per batch (bounded by the
retained output memory), then the measured batches; each batch is one
wall-clock interval over many operations whose outputs are retained until the
clock stops. Coverage (quick/full) comes from the versioned manifest
``benchmarks/cpu/manifests/perf_issues.yaml``; effort (scan/standard) is
independent. Counter diagnostics have no samples; their counters are reported
in their own table.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import yaml

from benchmark_layout import safe_target_profile
import bench_selection
import case_status as cs
import report_summary
from suite_instances import resolve_suite_instance_ids

ROOT = Path(__file__).resolve().parents[1]
# suite key -> (suite yaml, manifest, suite id)
SUITES = {
    "cpu": (ROOT / "benchmarks/cpu/perf_issues.yaml", ROOT / "benchmarks/cpu/manifests/perf_issues.yaml",
            "cpu/perf_issues"),
    "gpu": (ROOT / "benchmarks/gpu/perf_issues.yaml", ROOT / "benchmarks/gpu/manifests/perf_issues.yaml",
            "gpu/perf_issues"),
}
RETENTION_BUDGET_BYTES = 512 << 20


def plan(suite_key="cpu", threads=None):
    suite_path, manifest_path, _ = SUITES[suite_key]
    suite = yaml.safe_load(suite_path.read_text())
    cases = {c["id"]: c for c in json.loads((ROOT / suite["problems"]["source"]).read_text())}
    manifest = bench_selection.load_manifest(manifest_path)
    if set(manifest["coverage"]["full"]) != set(resolve_suite_instance_ids(suite_path)):
        raise bench_selection.SelectionError("perf_issues manifest full list != suite include list")
    coverage = bench_selection.resolve_coverage()
    expected, selected, raw = bench_selection.select(manifest, coverage)
    if threads is not None:
        eligible = lambda cid: threads in cases[cid].get("thread_counts", [threads])
        expected = [cid for cid in expected if eligible(cid)]
        selected = [cid for cid in selected if eligible(cid)]
    return suite, cases, manifest, coverage, expected, selected, raw


# ---------------------------------------------------------------------------
# PyTorch reference arms
# ---------------------------------------------------------------------------

def _values(n, seed):
    return [((((i * 2654435761) + seed * 97) % 2001) / 1000.0 - 1.0) * 0.5 for i in range(n)]


def torch_case(case):
    """Return (setup -> (op, check, retained_bytes), timer, outside) for a PyTorch arm."""
    import torch

    p = case["params"]
    kind = case["kind"]
    if kind.startswith("umbrella_"):
        from cpu_umbrella_references import torch_case as umbrella_torch_case
        return umbrella_torch_case(case)
    if kind == "cpu_gap_mwe":
        from cpu_gap_mwe_python import fixture
        op, validate, retained = fixture(p["operation"])
        def check():
            validate(op())
            return 0.0  # assertion checks against the analytical solution
        return op, check, retained, [p["operation"], "intrinsic output allocation", "Python dispatch"]
    if kind == "decode_projection":
        din, dout, length = p["in"], p["out"], p["len"]
        # Same logical x (in, len) and W (in, out) as the Rust arms; PyTorch's
        # native row-major Linear layout is x (len, in), W (out, in).
        x = torch.tensor(_values(din * length, 1), dtype=torch.float32).reshape(length, din)
        w = torch.tensor(_values(din * dout, 2), dtype=torch.float32).reshape(dout, din)

        def check():
            y = torch.nn.functional.linear(x, w)
            ref = x.double() @ w.double().T
            return float(((y.double() - ref).abs() / ref.abs().clamp(min=1.0)).max())
        return (lambda: torch.nn.functional.linear(x, w)), check, length * dout * 4, ["torch.nn.functional.linear", "output_allocation"]
    if kind == "decode_projection_view_operand":
        din, dout, length = p["in"], p["out"], p["len"]
        # Same logical x (in, len) and W (in, out) as the Rust arms; PyTorch
        # stores W as its native row-major [out, in] and reads it transposed.
        x = torch.tensor(_values(din * length, 1), dtype=torch.float32).reshape(length, din)
        w = torch.tensor(_values(din * dout, 2), dtype=torch.float32).reshape(dout, din)

        def check():
            y = x @ w.T
            ref = x.double() @ w.double().T
            return float(((y.double() - ref).abs() / ref.abs().clamp(min=1.0)).max())
        return (lambda: x @ w.T), check, length * dout * 4, ["torch.matmul(x, W.T)", "output_allocation"]
    if kind == "rank1_dot":
        n = p["len"]
        re_x, im_x, re_y, im_y = _values(n, 9), _values(n, 10), _values(n, 11), _values(n, 12)
        if case["dtype"] == "c64":
            x = torch.complex(torch.tensor(re_x, dtype=torch.float64), torch.tensor(im_x, dtype=torch.float64))
            y = torch.complex(torch.tensor(re_y, dtype=torch.float64), torch.tensor(im_y, dtype=torch.float64))
        else:
            x, y = torch.tensor(re_x, dtype=torch.float64), torch.tensor(re_y, dtype=torch.float64)

        def check():
            got = complex(torch.vdot(x, y))
            ref = complex((x.conj() * y).sum())
            return abs(got - ref) / max(abs(ref), 1.0)
        return (lambda: torch.vdot(x, y)), check, 64, ["torch.vdot (BLAS ddot/zdotc)", "scalar tensor allocation"]
    if kind == "small_solves":
        k, batch = p["k"], p["batch"]
        a = torch.zeros(batch, k, k, dtype=torch.float64)
        for b in range(batch):
            raw = torch.tensor(_values(k * k, 20 + b), dtype=torch.float64).reshape(k, k).T  # column-major item
            a[b] = torch.tril(raw / k, -1) + torch.eye(k, dtype=torch.float64)
        rhs = torch.stack([torch.tensor(_values(k * k, 500 + b), dtype=torch.float64).reshape(k, k).T for b in range(batch)])
        rhs = rhs.contiguous()
        if case["arm"] == "pytorch-solve-batched":
            op = lambda: torch.linalg.solve(a, rhs)
            label = "torch.linalg.solve on (batch, k, k)"
        else:
            op = lambda: torch.linalg.solve_triangular(a, rhs, upper=False, unitriangular=True)
            label = "torch.linalg.solve_triangular on (batch, k, k)"

        def check():
            x = op()
            return float(((a @ x - rhs).abs() / rhs.abs().clamp(min=1.0)).max())
        return op, check, batch * k * k * 8, [label, "output_allocation"]
    if kind == "composed_norm":
        d, length, batch = p["d"], p["len"], p["batch"]
        # Feature-first (d, len, batch) column-major == row-major (batch, len, d).
        x = torch.tensor(_values(d * length * batch, 30), dtype=torch.float32).reshape(batch, length, d)
        w = 1.0 + torch.tensor(_values(d, 31), dtype=torch.float32)
        bias = torch.tensor(_values(d, 32), dtype=torch.float32)
        if p["norm"] == "layer_norm":
            op = lambda: torch.nn.functional.layer_norm(x, (d,), w, bias, 1e-5)

            def reference():
                xd = x.double()
                mean = xd.mean(-1, keepdim=True)
                var = ((xd - mean) ** 2).mean(-1, keepdim=True)
                return (xd - mean) / torch.sqrt(var + 1e-5) * w.double() + bias.double()
        else:
            op = lambda: torch.nn.functional.rms_norm(x, (d,), w, 1e-5)

            def reference():
                xd = x.double()
                return xd / torch.sqrt((xd ** 2).mean(-1, keepdim=True) + 1e-5) * w.double()

        def check():
            ref = reference()
            return float(((op().double() - ref).abs() / ref.abs().clamp(min=1.0)).max())
        return op, check, d * length * batch * 4, [f"torch.nn.functional.{p['norm']} (fused)", "output_allocation"]
    # #2010 PR-B1 ops (#1975, #1976, #2008): same data as the Rust arms.
    if kind == "activation":
        rows, cols, name = p["rows"], p["cols"], p["op"]
        x = 8.0 * torch.tensor(_values(rows * cols, 50), dtype=torch.float32)
        f = torch.nn.functional
        ops = {"erf": torch.erf, "sigmoid": torch.sigmoid, "silu": f.silu, "softplus": f.softplus,
               "gelu": f.gelu, "gelu_tanh": lambda t: f.gelu(t, approximate="tanh")}
        op_fn = ops[name]

        def check():
            ref = op_fn(x.double())
            return float(((op_fn(x).double() - ref).abs() / ref.abs().clamp(min=1.0)).max())
        return (lambda: op_fn(x)), check, rows * cols * 4, [f"torch {name}", "output_allocation"]
    if kind == "softmax_pool":
        rows, cols = p["rows"], p["cols"]
        x = torch.tensor(_values(rows * cols, 60), dtype=torch.float64).reshape(cols, rows)
        def check():
            y = torch.softmax(x, -1)
            ref = (x - x.max(dim=-1, keepdim=True).values).exp()
            ref = ref / ref.sum(dim=-1, keepdim=True)
            return float((y - ref).abs().max())
        return (lambda: torch.softmax(x, -1)), check, rows * cols * 8, ["torch softmax", "output_allocation"]
    if kind == "softmax":
        length, batch, name = p["len"], p["batch"], p["op"]
        # Column-major (key, query, batch) == row-major (batch, query, key); reduce the key axis.
        x = 8.0 * torch.tensor(_values(length * length * batch, 60), dtype=torch.float32)
        x = x.reshape(batch, length, length)
        causal = torch.arange(length).reshape(1, length) <= torch.arange(length).reshape(length, 1)
        if name == "softmax":
            op_fn = lambda t: torch.softmax(t, -1)
        elif name == "log_softmax":
            op_fn = lambda t: torch.log_softmax(t, -1)
        else:
            op_fn = lambda t: torch.softmax(t.masked_fill(~causal, float("-inf")), -1)

        def check():
            ref = op_fn(x.double())
            return float(((op_fn(x).double() - ref).abs() / ref.abs().clamp(min=1.0)).max())
        return (lambda: op_fn(x)), check, length * length * batch * 4, [f"torch {name}", "output_allocation"]
    if kind == "reduce_mean":
        rows, cols = p["rows"], p["cols"]
        # Column-major (rows, cols) == row-major (cols, rows); axis 0 is the last torch dim.
        x = torch.tensor(_values(rows * cols, 70), dtype=torch.float32).reshape(cols, rows)

        def check():
            ref = x.double().mean(-1)
            return float(((x.mean(-1).double() - ref).abs() / ref.abs().clamp(min=1.0)).max())
        return (lambda: x.mean(-1)), check, cols * 4, ["torch.mean", "output_allocation"]
    if kind == "small_grad_trace":
        # #2186: the Rust arm's matrix, rhs and finite-difference direction (column-major).
        op_name, n = p["op"], p["n"]
        symmetric = op_name == "eigh"

        def symmetrize(flat):
            if symmetric:
                for col in range(n):
                    for row in range(col):
                        mean = 0.5 * (flat[row + n * col] + flat[col + n * row])
                        flat[row + n * col] = mean
                        flat[col + n * row] = mean
            return flat

        matrix = [0.1 * x for x in _values(n * n, 4)]
        symmetrize(matrix)
        for i in range(n):
            matrix[i + n * i] += 2.0 + i / n
        direction = symmetrize(_values(n * n, 99))

        def col_major(flat):
            return torch.tensor(flat, dtype=torch.float64).reshape(n, n).T.contiguous()

        a0 = col_major(matrix)
        b0 = col_major(_values(n * n, 77))
        e0 = col_major(direction)

        def loss(a):
            if op_name == "qr":
                q, r = torch.linalg.qr(a)
                return q.sum() + r.sum()
            if op_name == "eigh":
                w, v = torch.linalg.eigh(a)
                return w.sum() + v.sum()
            if op_name == "lu":
                _pm, low, up = torch.linalg.lu(a)
                return low.sum() + up.sum()
            if op_name == "svd":
                u, sv, vh = torch.linalg.svd(a)
                return u.sum() + sv.sum() + vh.sum()
            return torch.linalg.solve(a, b0).sum()

        leaf = a0.clone().requires_grad_(True)

        def grad_once():
            return torch.autograd.grad(loss(leaf), leaf)[0]

        def check():
            # Fifth-order five-point stencil: truncation and rounding both stay far below 1e-10.
            analytic = float((grad_once() * e0).sum())
            step = 1e-4
            with torch.no_grad():
                f = {k: float(loss(a0 + k * step * e0)) for k in (-2, -1, 1, 2)}
            fd = (-f[2] + 8 * f[1] - 8 * f[-1] + f[-2]) / (12 * step)
            return abs(analytic - fd) / max(abs(fd), 1.0)
        return grad_once, check, 2 * n * n * 8, [f"torch.linalg.{op_name} + autograd.grad", "output_allocation"]
    if kind == "fft_axis_c2c":
        # #1765: the Rust arm's complex fixture (re = values(seed 3), im = values(seed 4)),
        # in column-major order; axis 0 is the transform axis.
        n = p["n"]
        re, im = _values(n ** 3, 3), _values(n ** 3, 4)
        flat = [complex(r, i) for r, i in zip(re, im)]
        x = torch.tensor(flat, dtype=torch.complex128).reshape(n, n, n).permute(2, 1, 0).contiguous()

        def check():
            got = torch.fft.fft(x, dim=0)
            ref = torch.stack([torch.fft.fft(x[:, j, l]) for j, l in [(0, 0), (n - 1, n // 2)]])
            return float((got[:, 0, 0] - ref[0]).abs().max() / ref[0].abs().max())
        return (lambda: torch.fft.fft(x, dim=0)), check, n ** 3 * 16, ["torch.fft.fft(dim=0)", "output_allocation"]
    if kind == "degenerate_dot_general":
        # #2111: the Rust arm's fixtures (seeds 50-55), reshaped to the logical operands.
        form, n = p["form"], p["n"]
        if form == "hadamard":
            a = torch.tensor(_values(n * n, 50), dtype=torch.float64).reshape(n, n).T
            b = torch.tensor(_values(n * n, 51), dtype=torch.float64).reshape(n, n).T
            op = lambda: torch.mul(a, b)
        elif form == "outer":
            a = torch.tensor(_values(n, 52), dtype=torch.float64)
            b = torch.tensor(_values(n, 53), dtype=torch.float64)
            op = lambda: torch.outer(a, b)
        elif form == "rank0":
            a = torch.tensor(_values(1, 54), dtype=torch.float64)[0]
            b = torch.tensor(_values(n * n, 55), dtype=torch.float64).reshape(n, n).T
            op = lambda: a * b
        else:
            raise ValueError(f"unknown degenerate dot_general form {form}")

        def check():
            return 0.0  # the Rust arm checks the values; this arm is timing only
        return op, check, n * n * 8, [f"torch {form} (degenerate dot)", "output_allocation"]
    if kind == "erased_binary_add":
        # #1766: the Rust arm's fixtures (seeds 60 and 61).
        n = p["n"]
        a = torch.tensor(_values(n, 60), dtype=torch.float64)
        b = torch.tensor(_values(n, 61), dtype=torch.float64)
        return (lambda: torch.add(a, b)), (lambda: 0.0), n * 8, ["torch.add", "output_allocation"]
    if kind == "eager_small_call":
        # #1704 / #1667 / #1904: the Rust arm's fixtures (seeds 70-74); tiny PyTorch ops.
        op_name = p["op"]
        if op_name == "add":
            x = torch.tensor(_values(8, 70), dtype=torch.float64)
            y = torch.tensor(_values(8, 71), dtype=torch.float64)
            return (lambda: torch.add(x, y)), (lambda: 0.0), 8 * 8, ["torch.add (8 elements)", "output_allocation"]
        if op_name == "gemm":
            a = torch.tensor(_values(16, 73), dtype=torch.float64).reshape(4, 4).T
            b = torch.tensor(_values(16, 74), dtype=torch.float64).reshape(4, 4).T
            return (lambda: torch.matmul(a, b)), (lambda: 0.0), 16 * 8, ["torch.matmul (4x4)", "output_allocation"]
        return None
    if kind == "take_along_axis":
        n, batch = p["n"], p["batch"]
        # Column-major x [n, n, batch] == row-major (batch, n_j, n_i); idx [n, 1, batch] == (batch, 1, n).
        x = torch.tensor(_values(n * n * batch, 80), dtype=torch.float64).reshape(batch, n, n)
        idx = torch.tensor([((2 * (b % (n // 2)) + 1) * i + (b * 7) % n) % n
                            for b in range(batch) for i in range(n)], dtype=torch.int64).reshape(batch, 1, n)

        def check():
            got = torch.take_along_dim(x, idx, dim=2)
            ref = torch.stack([x[b][:, idx[b, 0]] for b in range(batch)])
            return float((got - ref).abs().max())
        return (lambda: torch.take_along_dim(x, idx, dim=2)), check, n * n * batch * 8, \
            ["torch.take_along_dim", "output_allocation"]
    raise ValueError(f"no PyTorch arm for {kind}")


def run_torch(case, threads, config, correctness_only):
    import torch

    torch.set_num_threads(threads)
    op, check, retained, timer = torch_case(case)
    error = check()
    tol = 1e-4 if case["dtype"] == "f32" else 1e-10
    if case["kind"] == "small_grad_trace":
        # The check is a finite-difference gradient test; its noise floor is ~1e-10 at n=16
        # (eigh). The Rust arm checks the same gradient against 1e-5 with its own step.
        tol = 1e-8
    row = {"max_rel_error": error, "samples": [],
           "scope": {"timer": timer, "outside_timer": ["input_construction", "torch.set_num_threads", "correctness_check"]},
           "provider": torch.__config__.show().split("\n")[0].strip()}
    if case["kind"] == "cpu_gap_mwe":
        # The MWE validates analytically with assertions, rather than returning
        # a measured maximum error. Never report the sentinel as an exact zero.
        row["max_rel_error"] = None
        row["validation_method"] = "analytical solution assertions before sampling"
    if not (error <= tol):
        row.update(correctness_status="failed", error=f"relative error {error:e} exceeds {tol:e}")
        return row
    row["correctness_status"] = "passed"
    if correctness_only:
        return row
    for _ in range(max(config["warmups"], 1)):
        op()
    cap = max(1, min(1 << 20, RETENTION_BUDGET_BYTES // max(retained, 1)))
    target_ns = int(config["min_runtime_ms"] * 1_000_000)

    def batch(iterations):
        outputs = [None] * iterations
        start = time.perf_counter_ns()
        for i in range(iterations):
            outputs[i] = op()
        elapsed = time.perf_counter_ns() - start
        del outputs
        return elapsed

    iterations = 1
    while True:
        elapsed = batch(iterations)
        if elapsed >= target_ns or iterations >= cap:
            break
        iterations = min(iterations * 2, cap)
    row["calibration"] = {"target_ns": target_ns, "iterations": iterations, "elapsed_ns": elapsed}
    row["samples"] = [{"sample_index": i, "elapsed_ns": batch(iterations), "iterations": iterations}
                      for i in range(config["runs"])]
    return row


# ---------------------------------------------------------------------------
# Collection and report
# ---------------------------------------------------------------------------

def collect(binary, output, threads, correctness_only=False, suite_key="cpu", device=0):
    suite, all_cases, manifest, coverage, expected, selected, raw = plan(suite_key, threads)
    suite_id = SUITES[suite_key][2]
    config = dict(suite["defaults"]["run"])
    effort = bench_selection.resolve_effort(config, suite["defaults"].get("effort"))
    config.update(warmups=effort["warmups"], runs=effort["runs"])
    failed = False
    rows = []
    with output.open("w") as fh:
        for case in (all_cases[i] for i in selected):
            case_wall_start = time.monotonic_ns()
            row = dict(case, threads=threads, case_id=case["id"], samples=[],
                       case_key=cs.case_key(case["id"], threads), effort=effort["effort"])
            if case["kind"] == "cpu_followup_mwe" or "followup_mwe" in case:
                try:
                    from cpu_followup_cases import run as run_followup
                    row.update(run_followup(case, threads, config, correctness_only))
                except Exception as error:
                    row.update(correctness_status="failed", samples=[], error=repr(error))
            elif case["kind"] == "metadata_clone" and case["backend"] == "julia-base":
                try:
                    command = [os.environ.get("JULIA", "julia"), "--project=.",
                               str(ROOT / "scripts/benchmark_metadata_clone.jl"),
                               str(0 if correctness_only else config["runs"]),
                               str(int(config["min_runtime_ms"] * 1_000_000))]
                    result = subprocess.run(command, text=True, capture_output=True, check=True)
                    row.update(json.loads(result.stdout.strip().splitlines()[-1]))
                except Exception as error:
                    row.update(correctness_status="failed", samples=[], error=repr(error))
            elif case["backend"] == "pytorch-cuda":
                try:
                    from gpu_perf_issue_references import run as run_gpu_reference
                    row.update(run_gpu_reference(case, config, correctness_only, device))
                except Exception as error:
                    row.update(correctness_status="failed", samples=[], error=repr(error))
            elif case["backend"] == "pytorch-cpu":
                try:
                    row.update(run_torch(case, threads, config, correctness_only))
                except Exception as error:  # a crash is a failed row, never a latency
                    row.update(correctness_status="failed", error=repr(error))
            else:
                command = [str(binary), "--case", case["id"], "--params", json.dumps(case["params"]),
                           "--threads", str(threads), "--device", str(device),
                           "--warmups", str(config["warmups"]),
                           "--samples", str(config["runs"]),
                           "--target-ns", str(int(config["min_runtime_ms"] * 1_000_000))]
                if correctness_only:
                    command += ["--mode", "correctness-only"]
                result = subprocess.run(command, text=True, capture_output=True)
                if result.returncode:
                    row.update(correctness_status="failed", error=result.stderr.strip())
                else:
                    try:
                        row.update(json.loads(result.stdout))
                    except (ValueError, TypeError) as error:
                        row.update(correctness_status="failed", error=f"invalid case output: {error}")
            if suite_key == "gpu":
                row["case_process_and_validation_wall_ns"] = time.monotonic_ns() - case_wall_start
            row["status"] = row["correctness_status"]
            failed |= row["status"] not in ("passed", "unsupported")
            values = [s["elapsed_ns"] / s["iterations"] for s in row["samples"]]
            row["cov"] = cs.coefficient_of_variation(values) if row["status"] == "passed" else None
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows.append(row)
            print(f"{case['id']}: {row['correctness_status']}", flush=True)
    status = cs.build_case_status(
        suite_id=suite_id, manifest_version=manifest["manifest_version"], coverage=coverage,
        effort=effort["effort"], expected=[cs.case_key(i, threads) for i in expected],
        selected=[cs.case_key(i, threads) for i in selected], rows=rows, selection_filter=raw)
    cs.write_case_status(output.parent / f"case_status_t{threads}.json", status)
    return failed or bool(status["missing"])


def summary(row):
    if row["correctness_status"] == "unsupported":
        return "—", "—", "—", "UNSUPPORTED"
    if row["correctness_status"] != "passed":
        return "—", "—", "—", "FAILED"
    values = [s["elapsed_ns"] / s["iterations"] for s in row["samples"]]
    if not values:
        return "—", "—", "—", "MISSING"
    median = statistics.median(values)
    q = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [median] * 3
    cov = statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0.0
    return f"{median:.1f}", f"{q[2] - q[0]:.1f}", f"{cov * 100:.1f}%", "NOISY" if cov > 0.10 else "ok"


def shape_label(row):
    return ", ".join(f"{k}={v}" for k, v in row["params"].items() if k not in {"kind", "arm", "dtype"})


def _numeric_summary(row):
    if row.get("correctness_status") != "passed":
        return None, None
    values = [s["elapsed_ns"] / s["iterations"] for s in row.get("samples", [])]
    if not values:
        return None, None
    median = statistics.median(values)
    q = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [median] * 3
    return median, q[2] - q[0]


def build_summary(run_dir, target, suite_key="cpu"):
    """Machine-readable sidecar for the same records the report is built from."""
    rows = [json.loads(line) for path in sorted(run_dir.glob("samples_t*.jsonl"))
            for line in path.read_text().splitlines()]
    metadata = None
    for path in sorted(run_dir.glob("run_t*.yaml")):
        metadata = yaml.safe_load(path.read_text())
        break
    header = report_summary.metadata_from_run_metadata(metadata)
    hardware = dict(header["hardware"])
    if suite_key != "gpu":
        try:
            from collect_cpu_info import collect_cpu_info
            hardware.setdefault("cpu", collect_cpu_info().get("Model", "unknown"))
        except Exception:  # pragma: no cover - best-effort host label
            hardware.setdefault("cpu", "unknown")
    threads = sorted({int(row["threads"]) for row in rows if row.get("threads") is not None})
    if threads:
        hardware["threads"] = threads
    out_rows = []
    for row in rows:
        median, iqr = _numeric_summary(row)
        status = "ok" if row.get("correctness_status") == "passed" else str(row.get("correctness_status", "failed"))
        out_rows.append(
            report_summary.make_row(
                row["case_id"],
                row["backend"],
                median,
                status=status,
                iqr=iqr,
                group=row.get("measurement_scope"),
                verified=row.get("correctness_status") == "passed",
                reason=(row.get("unsupported_reason") or row.get("error")) if status != "ok" else None,
            )
        )
    return report_summary.build_document(
        out_rows,
        target_profile=header["target_profile"] or target,
        suite_id=header["suite_id"] or SUITES[suite_key][2],
        timestamp=header["timestamp"],
        tenferro_commit=header["tenferro_commit"],
        hardware=hardware,
        unit="ns/op",
    )


def report(run_dir, target, suite_key="cpu"):
    suite_id = SUITES[suite_key][2]
    run_dir = run_dir.resolve()
    rows = [json.loads(line) for path in sorted(run_dir.glob("samples_t*.jsonl"))
            for line in path.read_text().splitlines()]
    lines = [f"# {suite_key.upper()} Performance-Issue Workloads", "", f"- Suite: `{suite_id}`",
             f"- Raw run: `{run_dir.relative_to(ROOT)}`", ""]
    for info in ("cpu_info.md", "gpu_info.md"):
        if (run_dir / info).exists():
            lines += [(run_dir / info).read_text().rstrip(), ""]
    for path in sorted(run_dir.glob("run_t*.yaml")):
        metadata = yaml.safe_load(path.read_text())
        conditions = {k: metadata[k] for k in ("timestamp", "tenferro_rs", "blas", "environment") if k in metadata}
        lines += [f"## {path.stem}", "", f"Full metadata: `{path.relative_to(ROOT)}`", "",
                  "```yaml", yaml.safe_dump(conditions, sort_keys=False).rstrip(), "```", ""]
    parts = [json.loads(p.read_text()) for p in sorted(run_dir.glob("case_status_t*.json"))]
    if parts:
        status = cs.merge_case_status(parts)
        cs.write_case_status(run_dir / "case_status.json", status)
        lines += cs.markdown_section(status)
    passed = sum(r["correctness_status"] == "passed" for r in rows)
    lines += [f"Numerical checks: **{passed}/{len(rows)} recorded rows passed** (case × thread count).", ""]
    header = ["| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |",
              "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|"]

    def table(selected, sequence=False):
        out = list(header)
        if sequence:
            out[0] = out[0].replace("Ops/batch", "Sequences/sample").replace("Median ns/op", "Median ns/sequence")
        for row in selected:
            median, iqr, cov, status = summary(row)
            out.append("| " + " | ".join(map(str, [
                row["case_id"], " ".join(row["issues"]), row["arm"], row["backend"], row["dtype"],
                shape_label(row), row["threads"],
                row["samples"][0]["iterations"] if row["samples"] else "—", median, iqr, cov, status])) + " |")
        return out

    steady = [r for r in rows if r["measurement_scope"] == "steady_state"]
    allocation_policy = ("" if suite_key == "gpu" else
                         "faer-direct and host-sgemm write into a preallocated output, the tenferro and "
                         "PyTorch arms allocate their output. ")
    lines += ["## Steady-state operation rows", "",
              "Each batch is one wall-clock interval over many operations; every output is retained until the "
              "clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard "
              "deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, "
              "runtimes, session entry, tracing/compilation, preparation) is outside timing; "
              + allocation_policy +
              "`reference_for` in the case file names the case each reference arm is compared with. "
              "GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); "
              "the Threads column is the host thread setting.", ""]
    default_regime = yaml.safe_load(SUITES[suite_key][0].read_text())["defaults"].get("regime", "throughput")
    for regime, title in (("throughput", "Compute- and bandwidth-bound operations"),
                          ("latency", "Latency and per-call overhead")):
        selected = [r for r in steady if (r.get("regime") or default_regime) == regime]
        if selected:
            lines += [f"### {title}", ""]
            if regime == "latency":
                lines += ["These rows measure dispatch/operation latency; they do not measure compute throughput.", ""]
            lines += table(selected) + [""]
    for scope, title, note in (
            ("session_entry_diagnostic", "Per-call session entry diagnostics (not operation comparisons)",
             "The measured call includes session/scope entry or exit: the eager-per-call arms enter and "
             "leave an eager session per operation, the raw-scope arms enter and leave a raw CUDA scope "
             "(or time the explicit barrier they replace), and the vendor arms enter a raw scope per "
             "operation. Each case's timing note lists its own boundary."),
            ("eager_ad_workflow", "Eager AD workflow diagnostics",
             "Forward + reduction + backward through the eager runtime, including its internal session entries; "
             "gradients accumulate across a batch and are reset outside timing."),
            ("issue_workflow_diagnostic", "Issue-workload diagnostics (not steady-state operation rows)",
             "The workload of the issue needs a per-call precondition that cannot be batched (a cache miss "
             "that reconstructs the contraction plan and its workspace, or a caller-provided destination "
             "handle built per call), so a sample is exactly one call: the median is that call's latency, "
             "not an average over a batch, and the precondition sits outside its clock.")):
        selected = [r for r in rows if r["measurement_scope"] == scope]
        if selected:
            lines += ["", f"## {title}", "", note, ""] + table(selected)
    streams = [r for r in rows if r["measurement_scope"] == "shape_stream_diagnostic"]
    if streams:
        lines += ["", "## Whole-sequence CUDA shape-stream latency", "",
                  "One sample is the entire recorded sequence, including required planning and one final native "
                  "synchronization. Cache reset and explicitly declared key warmup are outside timing. "
                  "Median/IQR describe sequence latency, not single-operation throughput. Enqueue durations and "
                  "the exact key/shape/stride sequence are retained in raw rows. Counter observations run separately. "
                  "NOISY is descriptive; these screens establish no baseline/candidate performance claim.", ""]
        lines += table(streams, sequence=True)
        for row in streams:
            lines += ["", f"### {row['case_id']} sequence", "",
                      "```json", json.dumps(row.get("sequence", []), separators=(",", ":")), "```", ""]
    counters = [r for r in rows if r["measurement_scope"] == "counter_diagnostic"]
    if counters:
        lines += ["", "## Counter diagnostics (no timing)", ""]
        for row in counters:
            lines += [f"### {row['case_id']} ({' '.join(row['issues'])}, {row['threads']} threads)", ""]
            if row["correctness_status"] == "unsupported":
                lines += [f"UNSUPPORTED: {row.get('unsupported_reason', 'public counter unavailable')}", ""]
            elif row["correctness_status"] != "passed":
                lines += ["FAILED", ""]
                continue
            if row.get("kind") == "shape_stream":
                lines += ["Untimed public cache/workspace/retirement observations; cumulative counters. "
                          "Exact sequence and reset scope are recorded in the raw row.", "",
                          "| Phase | Index | Key | Plans | Hits | Misses | Evictions | Plan bytes | Workspace bytes | In flight |",
                          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
                for observation in row.get("counter_observations", []):
                    stats = observation["stats"]
                    plans = stats["plans"]
                    fields = [observation["phase"], observation.get("index", "—"), observation.get("key", "—"),
                              plans["entries"], plans["hits"], plans["misses"], plans["evictions"],
                              plans["retained_bytes"], stats["workspace"]["bytes"], stats["retirement"]["in_flight"]]
                    lines.append("| " + " | ".join(map(str, fields)) + " |")
                lines.append("")
                continue
            lines += [f"Source: {row.get('counter_source', '')}. Unavailable: {', '.join(row.get('unavailable', []))}.", "",
                      "| Op | Output bytes | Host allocations/op | Allocated bytes/op | Allocated/output |",
                      "|---|---:|---:|---:|---:|"]
            for c in row.get("counters", []):
                ratio = c['allocated_bytes_over_output_bytes']
                ratio_text = "—" if ratio is None else f"{ratio:.2f}"
                lines.append(f"| {c['op']} | {c['output_bytes']} | {c['host_allocations_per_op']:.2f} | "
                             f"{c['host_allocated_bytes_per_op']:.0f} | {ratio_text} |")
            for field in ("size_of_bytes", "diagnostic_timing", "descriptor_cache_before",
                          "descriptor_cache_after", "binary_plan_builds_per_call",
                          "requested_threads", "actual_lanes", "actual_item_threads", "operation_requires_session",
                          "output_retention", "session_entry", "fused_regions"):
                if field in row:
                    lines += ["", f"{field}: `{json.dumps(row[field], separators=(',', ':'))}`"]
            lines.append("")
    states = [r for r in rows if r["measurement_scope"] == "state_diagnostic"]
    if states:
        lines += ["", "## State diagnostics (no timing)", "",
                  "Allocation and lifetime state, read before and after each arm: driver free device "
                  "bytes for the pool, and process RSS where the issue's observable is host memory "
                  "(#1924). Bytes, not nanoseconds; a case that keeps memory it no longer needs shows "
                  "up as a non-zero retained or growth value, and a probe that the driver refuses "
                  "shows up in `probe_status`.", ""]
        skip = {"suite_id", "case_id", "kind", "arm", "backend", "dtype", "issues",
                "measurement_scope", "profiles", "intent", "params", "reference_for", "threads",
                "case_key", "effort", "correctness_status", "status", "max_rel_error", "samples",
                "scope", "cov", "error", "per_cycle", "id", "device_ordinal"}
        for row in states:
            lines += [f"### {row['case_id']} ({' '.join(row['issues'])})", ""]
            if row["correctness_status"] != "passed":
                lines += ["FAILED", ""]
                continue
            lines += ["| Field | Value | MiB |", "|---|---:|---:|"]
            for key, value in row.items():
                if key in skip or not isinstance(value, (int, float, str, bool)):
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    lines.append(f"| {key} | `{value}` | — |")
                elif "bytes" in key:
                    lines.append(f"| {key} | {value} | {float(value) / (1 << 20):.1f} |")
                else:
                    lines.append(f"| {key} | {value} | — |")
            lines.append("")
            if row.get("per_cycle"):
                columns = list(row["per_cycle"][0].keys())
                lines.append("| Cycle | " + " | ".join(columns) + " |")
                lines.append("|---:" + "|---:" * len(columns) + "|")
                for index, cycle in enumerate(row["per_cycle"]):
                    lines.append(f"| {index} | " + " | ".join(str(cycle[c]) for c in columns) + " |")
                lines.append("")
    first_calls = [r for r in rows if r["measurement_scope"] == "first_call_diagnostic"]
    if first_calls:
        lines += ["", "## First-call diagnostics (not steady-state rows)", "",
                  "One cold process; each distinct layout is uploaded outside timing, then its first and "
                  "second call are timed separately (each includes a device synchronize).", ""]
        for row in first_calls:
            lines += [f"### {row['case_id']} ({' '.join(row['issues'])})", ""]
            if row["correctness_status"] != "passed":
                lines += ["FAILED", ""]
                continue
            lines += [f"Median over {row['distinct_layouts']} distinct layouts: first call "
                      f"{row['median_first_call_ms']:.3f} ms, second call {row['median_second_call_ms']:.3f} ms.", "",
                      "| Layout | Warm-up layout | First call ms | Second call ms |", "|---|---|---:|---:|"]
            for c in row["first_call_layouts"]:
                lines.append(f"| {c['layout']} | {'yes' if c['warm_up_layout'] else ''} | "
                             f"{c['first_call_ms']:.3f} | {c['second_call_ms']:.3f} |")
            lines.append("")
    lines += ["", "## Timing boundaries", ""]
    # Key on the boundary, not just the case: one (kind, arm) can run under two
    # different boundaries (an upload and a download arm, an f64 and a c64 arm).
    scopes = {}
    for row in rows:
        if row.get("scope"):
            scope = row["scope"]
            key = (row["kind"], row["arm"], tuple(scope["timer"]), tuple(scope["outside_timer"]))
            scopes[key] = scope
    for (kind, arm, _, _), scope in sorted(scopes.items()):
        lines.append(f"- **{kind} / {arm}** — inside: {', '.join(scope['timer']) or '—'}; "
                     f"outside: {', '.join(scope['outside_timer'])}.")
    for row in rows:
        if row.get("error"):
            lines += ["", f"### FAILED: {row['case_id']} ({row['threads']} threads)", "```", row["error"], "```"]
    text = "\n".join(lines) + "\n"
    (run_dir / "report.md").write_text(text)
    if suite_key == "gpu" and os.environ.get("GPU_BENCH_SKIP_LATEST") == "1":
        print(run_dir / "report.md")
        return
    latest = ROOT / "result" / safe_target_profile(target) / f"{suite_id}.md"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(text)
    report_summary.write_document(
        report_summary.summary_path_for_report(latest),
        build_summary(run_dir, target, suite_key),
    )
    print(latest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--target-profile", default="amd-cpu")
    parser.add_argument("--suite", choices=sorted(SUITES), default="cpu")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.report:
        report(args.report, args.target_profile, args.suite)
    else:
        raise SystemExit(collect(args.binary, args.output, args.threads, args.correctness_only,
                                 args.suite, args.device))
