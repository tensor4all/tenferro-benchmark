#!/usr/bin/env python3
"""Render one recorded CPU run directory to its report Markdown and summary.

Usage:
    python scripts/render_run.py <run_dir> [--out DIR]

The renderer is a pure function of the run directory: it reads the recorded
``run.yaml``/``run_t<N>.yaml`` metadata, ``cpu_info.md``, ``thread_env.txt`` and
the raw measurement files, and never probes the current CPU, shell
environment or clock.  It dispatches on the run's suite and writes the same
report filenames collection uses (``report.md``, ``cpu_ops_report.md``, ...)
plus ``<suite>.summary.json`` into ``--out`` (default: the run directory).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import report_summary  # noqa: E402

ROOT = SCRIPT_DIR.parent

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OMP_DYNAMIC",
    "RAYON_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "VECLIB_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "XLA_FLAGS",
)

EINSUM_LOG_PREFIXES = (
    "tenferro_trace_",
    "tenferro_eager_",
    "pytorch_cpu_",
    "jax_cpu_",
    "julia_omeinsum_",
)

PUBLIC_API_TIMING_DISCIPLINE = (
    "- Input fixture tensors are created during warmup and outside the measured region for tenferro-rs, PyTorch, and JAX.",
    "- The tenferro-rs direct column measures immediate public operations (normally concrete `Tensor + CpuBackend`; `lstsq`/`svd_full` use `EagerTensor` because no concrete spelling exists). It is not labeled as the `EagerTensor` call layer.",
    "- tenferro-rs trace graphs are constructed and compiled outside the measured region; each compiled graph is reused for every warmup and timed run.",
    "- JAX functions are compiled with `jax.jit` during warmup, outside the measured region; timed calls include dispatch through `jax.block_until_ready`.",
    "- Allocation-returning API rows create their output tensor inside each timed call.",
    "- `cpu/output_reuse` rows allocate the destination during warmup and reuse it; tenferro-rs `*_into` is compared with PyTorch `out=`/`copy_`.",
    "- Trace mode is `unsupported` for caller-output rows because compiled tenferro-rs graphs own their output tensors; JAX is shown as missing because it has no equivalent mutable `out=` API.",
    "- PyTorch view-producing indexing operations are cloned inside the timed region to match tenferro-rs owned, materialized outputs.",
    "- `reshape` compares materialized outputs: PyTorch clones its reshape view inside timing to match tenferro-rs' output-sized write; JAX has value semantics and no public strided-view contract.",
    "- `cpu/view_metadata` separately compares concrete tenferro-rs, PyTorch, and Julia (`julia-base`) view creation without an output-sized copy; trace mode is unsupported and JAX is missing because neither exposes the same concrete strided-view contract.",
    "- Julia `cpu/view_metadata` rows are `reshape`/`transpose`/`@view` lazy wrappers with no output-sized copy; `broadcast_in_dim_view` has no natural Base spelling and stays missing for Julia.",
    "- Julia `cpu/output_reuse` rows use broadcast-into (`.=`), `mul!`, and `copyto!` with destinations allocated during warmup and reused, the same reuse discipline as the tenferro-rs `_into`/PyTorch `out=` rows above.",
    "- Rust, PyTorch, and JAX fixtures contain identical logical values. Python/JAX reconstruct the Rust column-major fixture in each framework's native layout before timing.",
    "- Compact Householder QR and RRQR use explicit fixture/session setup; QR extraction and reconstruction/orthogonality checks are untimed. RRQR retains Q, R, permutation and numerical rank; JAX's three-output pivoted QR and PyTorch's unpivoted QR are not equivalent and are unsupported. Compact trace state and caller-output trace solve are unsupported.",
    "- The gap cases use explicit setup and numerical validation outside sampling. Axis-wise sum-of-squares and PyTorch/Julia masked log-softmax compositions are excluded from direct operation comparisons.",
    "- PyTorch complex conjugation uses `torch.conj_physical` to match tenferro-rs physical output rather than the lazy conjugate view from `torch.conj`.",
    "- `dot_general_with_conj` instead uses PyTorch's lazy conjugate view so conjugation can be handled by the contraction, matching tenferro-rs' conjugation flags; trace is unsupported because there is no equivalent public traced API. Julia's `dot_general_with_conj` row materializes the conjugate before the GEMM instead, since Julia has no lazy conj-without-transpose spelling that BLAS can fuse.",
    "- `pad` has no natural Base spelling and stays missing for Julia.",
    "- `dynamic_update_slice` reports trace mode as `unsupported` because tenferro-rs does not currently expose a corresponding `TracedTensor` API.",
    "- `full_piv_lu` and `full_piv_lu_solve` are excluded because PyTorch has no direct public full-pivot equivalent; substituting `torch.linalg.solve` would compare different algorithms.",
    "- `cpu/linalg_batched` rows use a batch of 1024 matrices with one rhs column. `batched_lu_factor` times the packed LU factorization alone. `batched_lu_solve` reuses LU factors prepared during warmup, outside the measured region, and times only the solve (tenferro-rs `LinalgBackend::lu_solve_prepared`, PyTorch `torch.linalg.lu_solve`, JAX `jax.scipy.linalg.lu_solve`, Julia `LAPACK.getrs!`). `batched_triangular_solve` times a lower-triangular solve.",
    "- Trace mode is `unsupported` for `batched_lu_factor` and `batched_lu_solve` because `LinalgOp` is not public, so a trace cannot hold a bare LU factor or prepared solve. The traced solve (LU factor plus prepared solve) and its backward are measured by the `cpu/cpu_ops` `batched_solve` and `grad_sum_batched_solve_backward` rows.",
    "- Julia has no batched LAPACK entry point, so the Julia `cpu/linalg_batched` rows loop the per-matrix LAPACK call over the batch axis and allocate their outputs inside the timed call.",
    "- The tenferro-rs runner builds its direct `CpuBackend`, the trace runtime, and the eager runtime with exactly `--num-threads` workers, and it fails at startup when a thread environment variable disagrees with that count or when the backend execution scope reports a different Rayon thread count.",
    "- `svd_full` remains in the table even when the selected tenferro-rs provider reports it as unsupported.",
    "- Julia is column-major, like tenferro-rs, so the `julia-base`/`strided-jl` columns need no PyTorch/JAX-style layout reconstruction to keep the same logical fixture values.",
    "- Julia warmup runs move JIT compilation outside the measured region, the same way PyTorch/JAX warmups do.",
    "- Julia factorization rows materialize their factors inside the timed call (`cholesky` returns the factor matrix, `lu` returns P/L/U, `qr` returns Q/R). A Julia `Factorization` keeps its factors packed in LAPACK's working storage, so timing the compact object would compare strictly less work than the tenferro-rs and PyTorch columns, which return separate materialized tensors.",
    "- Julia dense linalg rows run through Julia's own BLAS/LAPACK (libblastrampoline, by default OpenBLAS), recorded as `julia.blas_provider` in the run metadata. When that differs from the provider tenferro-rs and PyTorch link against (Accelerate on macOS), those rows partly compare BLAS implementations rather than framework overhead; read them together with the recorded providers.",
    "- The Julia `lstsq` row uses `qr(a) \\ rhs` rather than `a \\ rhs`: the bare backslash runs a column-pivoted, rank-revealing QR (the LAPACK `gelsy` algorithm), while the PyTorch row selects the `gels` driver, so the unpivoted spelling is the like-for-like comparison.",
    "- `julia-base` uses the natural Base/LinearAlgebra spelling and `strided-jl` the natural Strided.jl (`@strided`) spelling; each is populated only for rows where that spelling naturally applies (reductions and dense linalg have no natural Strided.jl spelling, so `strided-jl` covers elementwise/chain/transpose rows, the elementwise `cpu/output_reuse` `_into` rows, and the elementwise `cpu/complex` rows conj/mul/div/exp/log).",
    "- Strided.jl (https://github.com/Jutho/Strided.jl) is prior art for tenferro-rs' strided-rs kernel layer; the `strided-jl` column credits that lineage directly in the report.",
)

FFT_TIMING_DISCIPLINE = (
    "- Input tensors are created outside the timed region.",
    "- Each timed call creates the output tensor.",
    "- tenferro-rs immediate rows require BENCH_INCLUDE_SETUP_DIAGNOSTICS=1 and use the separate cpu/fft_setup suite.",
    "- tenferro-rs read rows require BENCH_INCLUDE_SETUP_DIAGNOSTICS=1 because the API performs one-shot planning.",
    "- tenferro-rs cached rows reuse a caller-owned `FftExecutor` across warmups and timed runs.",
    "- tenferro-rs eager rows reuse one `EagerRuntime` and input `EagerTensor`; setup is outside timing.",
    "- tenferro-rs trace rows construct and compile `TracedTensorFftExt` graphs outside timing and reuse the compiled program.",
    "- PyTorch rows use `torch.fft` after warmup, allowing PyTorch internal planning/cache behavior.",
    "- The primary fair comparison is cached `FftExecutor` versus warmed `torch.fft`; one-shot diagnostics are opt-in and never operation comparisons.",
    "- This initial suite only measures 1D transforms to avoid row-major/column-major batched-axis layout artifacts.",
)


class RenderError(RuntimeError):
    """The run directory does not contain what its suite needs."""


# --------------------------------------------------------------------------
# CPU-ops raw samples
# --------------------------------------------------------------------------

# The publication gate keeps every batch timing it summarized; the CSV next to
# it holds only the derived statistics.
CPU_OPS_SAMPLE_GLOB = "cpu_ops_samples_t*.jsonl"
# Rows recorded by benchmark_cpu_ops_python.py, which derives its quartiles
# differently from publication_gate.rs (see verify_cpu_ops_samples).
PYTHON_CPU_OPS_BACKENDS = frozenset({"pytorch-cpu", "jax-cpu"})
# Both collectors print the statistics with `{:.6f}`.
CPU_OPS_STATISTIC_TOLERANCE_MS = 1e-6
# ponytail: the raw sample record carries neither its dtype nor the quartile
# convention that produced the CSV, so the row identity omits dtype (both
# collectors measure f64 only today) and the backend label selects the
# convention. A collector that records either fact makes both explicit.


def _sample_backend(backend: str) -> str:
    """The CSV backend name of one raw sample record."""
    # run_cpu_ops.sh renames the publication gate's provider rows to the
    # comparison column `tenferro-eager`.
    providers = {"native", "blas-openblas", "blas-accelerate", "blas-mkl", "cuda"}
    return "tenferro-eager" if backend in providers else backend


def _publication_gate_percentile(values: list[float], fraction: float) -> float:
    """`round((n - 1) * p)` of sorted values, as publication_gate.rs rounds it."""
    if not values:
        return 0.0
    return values[int(math.floor((len(values) - 1) * fraction + 0.5))]


def verify_cpu_ops_samples(run_dir: Path, csvs: list[Path]) -> None:
    """Re-derive the published CPU-ops statistics from the raw sample records.

    ``publication_gate.rs`` takes the median and the inclusive IQR of the batch
    times and divides by the operations per sample;
    ``benchmark_cpu_ops_python.py`` takes the median of the per-operation times
    but ``len // 4`` / ``3 * len // 4`` for its quartiles.  Both rules are
    replayed here for the backends that recorded the samples, so editing
    ``batch_elapsed_ns``, ``per_operation_ns``, ``median_ms`` or ``iqr_ms``
    fails the render even though the CSV alone would look consistent.  A
    record's columns are compared against the rounded values the CSV stores.
    """
    run_dir = Path(run_dir)
    records: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    sample_files = sorted(run_dir.glob(CPU_OPS_SAMPLE_GLOB))
    for path in sample_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") != "samples":
                continue
            # The sample record carries no dtype; both collectors record f64
            # rows only, so the identity below is unique per measurement.
            key = (
                str(record.get("suite")),
                str(record.get("benchmark")),
                str(record.get("shape")),
                _sample_backend(str(record.get("backend"))),
                int(record["threads"]),
            )
            if key in records:
                raise RenderError(f"{path}: duplicate raw CPU-ops sample record for {key}")
            records[key] = record
    if not records:
        return
    verified: set[tuple[str, str, str, str, int]] = set()
    for path in csvs:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if not (row.get("threads") or "").strip().isdigit():
                    continue
                key = (
                    row.get("suite", ""),
                    row.get("benchmark", ""),
                    row.get("shape", ""),
                    row.get("backend", ""),
                    int(row["threads"]),
                )
                record = records.get(key)
                if record is None:
                    continue
                verified.add(key)
                operations = int(record["operations_per_sample"])
                batch_ns = [float(value) for value in record["batch_elapsed_ns"]]
                per_operation_ns = [float(value) for value in record["per_operation_ns"]]
                if len(batch_ns) != len(per_operation_ns):
                    raise RenderError(
                        f"{path}: {key} records {len(batch_ns)} batch times "
                        f"but {len(per_operation_ns)} per-operation times"
                    )
                for batch_value, per_operation in zip(batch_ns, per_operation_ns):
                    if not math.isclose(per_operation, batch_value / operations, rel_tol=1e-9):
                        raise RenderError(
                            f"{path}: {key} per_operation_ns {per_operation} disagrees with "
                            f"batch_elapsed_ns {batch_value} / {operations} operations"
                        )
                ordered = sorted(value / operations for value in batch_ns)
                if key[3] in PYTHON_CPU_OPS_BACKENDS:
                    per_operation = sorted(per_operation_ns)
                    median = statistics.median(per_operation) / 1e6
                    iqr = (per_operation[(3 * len(per_operation)) // 4]
                           - per_operation[len(per_operation) // 4]) / 1e6
                else:
                    median = _publication_gate_percentile(ordered, 0.5) / 1e6
                    iqr = (_publication_gate_percentile(ordered, 0.75)
                           - _publication_gate_percentile(ordered, 0.25)) / 1e6
                if (row.get("status") or "") != "ok":
                    continue
                for column, expected in (("median_ms", median), ("iqr_ms", iqr)):
                    recorded = (row.get(column) or "").strip()
                    if not recorded:
                        continue
                    if abs(float(recorded) - expected) > CPU_OPS_STATISTIC_TOLERANCE_MS:
                        raise RenderError(
                            f"{path}: {key} {column} is {recorded}, but its raw samples "
                            f"derive {expected:.9f}"
                        )
    unrecorded = sorted(set(records) - verified, key=str)
    if unrecorded:
        raise RenderError(f"{run_dir}: raw CPU-ops samples without a CSV row: {unrecorded}")


# --------------------------------------------------------------------------
# Recorded metadata helpers
# --------------------------------------------------------------------------


def load_metadata(run_dir: Path) -> tuple[dict[str, Any], Path | None]:
    """The run's canonical metadata block, preferring ``run.yaml``."""
    run_dir = Path(run_dir)
    for name in ("run.yaml",):
        path = run_dir / name
        if path.is_file():
            return yaml.safe_load(path.read_text()) or {}, path
    thread_files = sorted(run_dir.glob("run_t*.yaml"))
    if thread_files:
        return yaml.safe_load(thread_files[0].read_text()) or {}, thread_files[0]
    return {}, None


def recorded_threads(run_dir: Path, metadata: dict[str, Any]) -> list[int]:
    threads = {
        int(match.group(1))
        for path in Path(run_dir).glob("run_t*.yaml")
        if (match := re.fullmatch(r"run_t(\d+)", path.stem))
    }
    threads.update(report_summary.recorded_threads(metadata))
    return sorted(threads)


def run_label(run_dir: Path) -> str:
    return report_summary.repo_relative(Path(run_dir).resolve())


def cpu_info_section(run_dir: Path) -> list[str]:
    path = Path(run_dir) / "cpu_info.md"
    if path.is_file():
        return [path.read_text().rstrip(), ""]
    return ["## CPU Information", "", "- unavailable: this run did not record `cpu_info.md`.", ""]


def thread_env_section(run_dir: Path) -> list[str]:
    """Thread environment recorded by collection, never the current shell."""
    run_dir = Path(run_dir)
    snapshot = run_dir / "thread_env.txt"
    if snapshot.is_file():
        return [snapshot.read_text().rstrip(), ""]
    lines = ["## Thread Environments", ""]
    for path in sorted(run_dir.glob("run_t*.yaml")):
        metadata = yaml.safe_load(path.read_text()) or {}
        environment = (metadata.get("environment") or {}).get("env") or {}
        match = re.fullmatch(r"run_t(\d+)", path.stem)
        lines += [f"### Threads: {match.group(1) if match else path.stem}", ""]
        for key in THREAD_ENV_KEYS:
            lines.append(f"- {key}: `{environment.get(key, '')}`")
        lines.append("")
    return lines


def blas_section(metadata: dict[str, Any]) -> list[str]:
    features = (metadata.get("tenferro_rs") or {}).get("features") or []
    blas = metadata.get("blas") or {}
    lines = ["## Tenferro CPU BLAS Backend", "", f"- tenferro-rs features: `{','.join(features)}`"]
    for key, label in (
        ("implementation", "BLAS implementation"),
        ("version", "BLAS version"),
        ("root", "BLAS root"),
        ("library", "BLAS library"),
    ):
        if blas.get(key):
            lines.append(f"- {label}: `{blas[key]}`")
    return lines + [""]


def python_backend_section(metadata: dict[str, Any]) -> list[str]:
    lines = ["## Python Backend Providers", ""]
    backends = metadata.get("python_backends") or {}
    if not backends:
        return lines + ["- unavailable: provider metadata was not collected", ""]
    for key, label in (("pytorch", "PyTorch"), ("jax", "JAX")):
        info = backends.get(key) or {}
        if not info.get("available"):
            lines.append(f"- {label}: unavailable ({info.get('reason') or 'unavailable'})")
            continue
        provider = info.get("provider") or "unknown"
        version = info.get("version") or "unknown"
        if key == "jax":
            details = [f"dot backend `{info.get('dot_backend') or provider}`", f"version `{version}`"]
            if info.get("jaxlib_version"):
                details.append(f"jaxlib `{info['jaxlib_version']}`")
            if info.get("backend"):
                details.append(f"default backend `{info['backend']}`")
            if info.get("lapack_provider"):
                details.append(f"LAPACK provider `{info['lapack_provider']}`")
        else:
            details = [f"BLAS provider `{provider}`", f"version `{version}`"]
            if info.get("blas_info"):
                details.append(f"BLAS_INFO `{info['blas_info']}`")
            if info.get("lapack_info"):
                details.append(f"LAPACK_INFO `{info['lapack_info']}`")
            if info.get("backend"):
                details.append(f"backend `{info['backend']}`")
        lines.append(f"- {label}: " + ", ".join(details))
        if info.get("linked_libraries"):
            dep_label = "linked LAPACK libs" if key == "jax" else "linked BLAS/LAPACK libs"
            lines.append(f"  - {dep_label}: " + "; ".join(f"`{dep}`" for dep in info["linked_libraries"]))
    return lines + [""]


def julia_section(metadata: dict[str, Any]) -> list[str]:
    lines = ["## Julia / OMEinsum.jl Backend", ""]
    julia = metadata.get("julia") or {}
    if not julia.get("available"):
        return lines + [f"- Julia: unavailable ({julia.get('reason') or 'unavailable'})", ""]
    details = [f"version `{julia.get('version') or 'unknown'}`"]
    if julia.get("omeinsum_version"):
        details.append(f"OMEinsum.jl `{julia['omeinsum_version']}`")
    if julia.get("blas_provider"):
        details.append(f"BLAS provider `{julia['blas_provider']}`")
    if julia.get("threads") is not None:
        details.append(f"probe threads `{julia['threads']}`")
    lines.append("- Julia: " + ", ".join(details))
    lines += [
        "",
        "- `omeinsum-jl` (mode `omeinsum_path` in the log/report) always",
        "  executes the instance's precomputed `opt_flops`/`opt_size` path via",
        "  `OMEinsum.DynamicEinCode` pairwise contractions; OMEinsum's own",
        "  contraction-order optimizer is never invoked, so the comparison",
        "  against tenferro/PyTorch/JAX (which also use the precomputed path)",
        "  stays fair.",
        "- `JULIA_NUM_THREADS` also pins `LinearAlgebra.BLAS.set_num_threads`,",
        "  matching the BLAS thread pinning used by the other CPU backends.",
        "- Julia is column-major like tenferro-rs, so the einsum runner uses",
        "  `format_string_colmajor` / `shapes_colmajor` directly with no",
        "  PyTorch/JAX-style layout reconstruction.",
        "- OMEinsum dispatches its pairwise contractions to Julia's own BLAS",
        "  (libblastrampoline, OpenBLAS by default; the provider that ran is",
        "  reported above). When that differs from the provider tenferro-rs and",
        "  PyTorch link against, matmul-shaped rows partly compare BLAS",
        "  implementations rather than einsum-runtime overhead; read them",
        "  together with the recorded providers. `docs/einsum-suite.md` records",
        "  the measured OMEinsum-over-its-own-BLAS overhead.",
        "",
    ]
    return lines


def run_header(
    *,
    title: str,
    suite_id: str,
    target: str,
    suite_file: str,
    metadata: dict[str, Any],
    run_dir: Path,
    run_metadata_name: str | None,
    threads: list[int],
    command: str | None = None,
    intro: list[str] | None = None,
) -> list[str]:
    timestamp = metadata.get("timestamp") or Path(run_dir).name
    lines = [f"# {title}", "", f"- Suite: `{suite_id}`", f"- Target profile: `{target}`"]
    if suite_file:
        lines.append(f"- Suite file: `{suite_file}`")
    if run_metadata_name:
        lines.append(f"- Run metadata: `{run_metadata_name}`")
    lines += [f"- Timestamp: `{timestamp}`", ""]
    threads_text = " ".join(str(thread) for thread in threads) if threads else "<threads>"
    lines.append(f"Latest run: `./scripts/{command or 'run_all.sh'} {threads_text}`.")
    lines += [""] + list(intro or [])
    commit = (metadata.get("tenferro_rs") or {}).get("commit")
    if commit:
        lines += [f"- tenferro-rs commit: `{commit}`", ""]
    return lines


def relative_artifact(run_dir: Path, path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# Suite renderers
# --------------------------------------------------------------------------


def render_einsum(run_dir: Path, out_dir: Path) -> list[Path]:
    import format_results

    logs = [
        path
        for path in sorted(Path(run_dir).glob("*.log"))
        if path.name.startswith(EINSUM_LOG_PREFIXES) and path.stat().st_size > 0
    ]
    if not logs:
        print(f"render_run: {run_dir}: no einsum logs; skipping the einsum report", file=sys.stderr)
        return []
    metadata, metadata_path = load_metadata(run_dir)
    target = metadata.get("target_profile") or "unknown"
    threads = recorded_threads(run_dir, metadata)
    results: dict[tuple[str, str, str], tuple[float | None, float | None]] = {}
    environment: dict[str, str] = {}
    for path in logs:
        parsed, log_environment = format_results.parse_log(str(path))
        results.update(parsed)
        environment.update(log_environment)
    table = format_results.format_markdown_table(results, environment)
    cpu_model = report_summary.cpu_model_from_run_dir(run_dir)
    summary = format_results.build_summary(
        results, environment, metadata, run_dir=run_dir, cpu_model=cpu_model
    )
    lines = run_header(
        title="Einsum Benchmark Results",
        suite_id="cpu/einsum",
        target=target,
        suite_file=metadata.get("suite_file", ""),
        metadata=metadata,
        run_dir=run_dir,
        run_metadata_name=relative_artifact(run_dir, metadata_path) if metadata_path else None,
        threads=threads,
        command="run_all.sh",
        intro=[
            f"This file is generated from one suite run under `{run_label(run_dir)}`.",
            "",
        ],
    )
    lines += cpu_info_section(run_dir) + thread_env_section(run_dir)
    lines += blas_section(metadata) + python_backend_section(metadata) + julia_section(metadata)
    lines += [f"## Threads: {' '.join(str(thread) for thread in threads) or 'unknown'}", ""]
    lines += ["Logs:", ""]
    for log in logs:
        lines.append(f"- `{relative_artifact(run_dir, log)}`")
    lines += ["", table.rstrip(), ""]
    return write_outputs(run_dir, out_dir, "report.md", "einsum", summary, lines)


def render_cpu_ops(run_dir: Path, out_dir: Path) -> list[Path]:
    import format_cpu_ops_results

    csvs = sorted(Path(run_dir).glob("cpu_ops_t*.csv"))
    if not csvs:
        return []
    verify_cpu_ops_samples(run_dir, csvs)
    metadata, metadata_path = load_metadata(run_dir)
    target = metadata.get("target_profile") or "unknown"
    threads = recorded_threads(run_dir, metadata)
    table = format_cpu_ops_results.format_table(csvs)
    cpu_model = report_summary.cpu_model_from_run_dir(run_dir)
    summary = format_cpu_ops_results.build_summary(
        csvs, metadata, suite_id="cpu/cpu_ops", run_dir=run_dir, cpu_model=cpu_model
    )
    lines = run_header(
        title="CPU Benchmark Results",
        suite_id="cpu/cpu_ops",
        target=target,
        suite_file=metadata.get("suite_file", ""),
        metadata=metadata,
        run_dir=run_dir,
        run_metadata_name=relative_artifact(run_dir, metadata_path) if metadata_path else None,
        threads=threads,
        command="run_all.sh",
        intro=[
            f"This file is generated from one CPU ops run under `{run_label(run_dir)}`.",
            "",
        ],
    )
    lines += cpu_info_section(run_dir) + thread_env_section(run_dir)
    lines += blas_section(metadata) + python_backend_section(metadata)
    lines += [f"## Threads: {' '.join(str(thread) for thread in threads) or 'unknown'}", ""]
    for csv_path in csvs:
        lines.append(f"- CSV: `{relative_artifact(run_dir, csv_path)}`")
    lines += ["", table.rstrip(), ""]
    return write_outputs(run_dir, out_dir, "cpu_ops_report.md", "cpu_ops", summary, lines)


def render_linalg_jvp_vjp(run_dir: Path, out_dir: Path) -> list[Path]:
    csvs = sorted(Path(run_dir).glob("cpu_ops_t*.csv"))
    selected = [
        path
        for path in csvs
        if re.search(r"_jvp,|_vjp,", path.read_text(errors="replace"))
    ]
    if not selected:
        return []
    verify_cpu_ops_samples(run_dir, csvs)
    import format_linalg_ad_results

    metadata, metadata_path = load_metadata(run_dir)
    target = metadata.get("target_profile") or "unknown"
    threads = recorded_threads(run_dir, metadata)
    table = format_linalg_ad_results.format_table(selected[0])
    cpu_model = report_summary.cpu_model_from_run_dir(run_dir)
    summary = format_linalg_ad_results.build_summary(
        selected[0], metadata, suite_id="cpu/linalg_jvp_vjp", run_dir=run_dir, cpu_model=cpu_model
    )
    lines = run_header(
        title="CPU Linalg JVP/VJP Benchmark Results",
        suite_id="cpu/linalg_jvp_vjp",
        target=target,
        suite_file=metadata.get("suite_file", ""),
        metadata=metadata,
        run_dir=run_dir,
        run_metadata_name=relative_artifact(run_dir, metadata_path) if metadata_path else None,
        threads=threads,
        command="run_all.sh",
        intro=[f"Derived from the CPU ops CSV under `{run_label(run_dir)}`.", ""],
    )
    lines += cpu_info_section(run_dir) + thread_env_section(run_dir)
    lines += blas_section(metadata) + python_backend_section(metadata)
    lines += [f"## Threads: {' '.join(str(thread) for thread in threads) or 'unknown'}", ""]
    for csv_path in selected:
        lines.append(f"- CSV: `{relative_artifact(run_dir, csv_path)}`")
    lines += ["", table.rstrip(), ""]
    return write_outputs(
        run_dir, out_dir, "linalg_jvp_vjp_report.md", "linalg_jvp_vjp", summary, lines
    )


def render_public_api(run_dir: Path, out_dir: Path) -> list[Path]:
    import format_cpu_ops_results

    csvs = sorted(Path(run_dir).glob("cpu_public_api_t*.csv"))
    if not csvs:
        raise RenderError(f"{run_dir}: no cpu_public_api CSVs")
    metadata, metadata_path = load_metadata(run_dir)
    target = metadata.get("target_profile") or "unknown"
    threads = recorded_threads(run_dir, metadata)
    table = format_cpu_ops_results.format_table(csvs)
    cpu_model = report_summary.cpu_model_from_run_dir(run_dir)
    suite_id = metadata.get("suite_id") or "cpu/public_api"
    summary = format_cpu_ops_results.build_summary(
        csvs, metadata, suite_id=suite_id, run_dir=run_dir, cpu_model=cpu_model
    )
    name = suite_id.split("/")[-1]
    lines = run_header(
        title="CPU Public API Benchmark Results",
        suite_id=suite_id,
        target=target,
        suite_file=metadata.get("suite_file", ""),
        metadata=metadata,
        run_dir=run_dir,
        run_metadata_name=relative_artifact(run_dir, metadata_path) if metadata_path else None,
        threads=threads,
        command="run_cpu_public_api.sh",
        intro=[
            "- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`",
            "",
            f"This file is generated from sequential CPU public API runs under `{run_label(run_dir)}`.",
            "",
        ],
    )
    sampling = metadata.get("collection") or {}
    if sampling.get("profile"):
        lines += [
            f"- Sampling: `{sampling['profile']}` profile, {sampling.get('measured_runs', '?')} "
            f"measured runs, {sampling.get('warmups', '?')} warmups per row",
            "",
        ]
    lines += cpu_info_section(run_dir) + thread_env_section(run_dir)
    lines += ["## Timing Discipline", ""] + list(PUBLIC_API_TIMING_DISCIPLINE) + [""]
    lines += [f"## Threads: {' '.join(str(thread) for thread in threads) or 'unknown'}", ""]
    for csv_path in csvs:
        lines.append(f"- CSV: `{relative_artifact(run_dir, csv_path)}`")
    lines += ["", table.rstrip(), ""]
    return write_outputs(run_dir, out_dir, "report.md", name, summary, lines)


def render_fft(run_dir: Path, out_dir: Path) -> list[Path]:
    import format_cpu_fft_results
    import format_cpu_ops_results

    csvs = sorted(Path(run_dir).glob("cpu_fft_t*.csv"))
    if not csvs:
        raise RenderError(f"{run_dir}: no cpu_fft CSVs")
    metadata, metadata_path = load_metadata(run_dir)
    target = metadata.get("target_profile") or "unknown"
    threads = recorded_threads(run_dir, metadata)
    table = format_cpu_fft_results.format_table(csvs)
    cpu_model = report_summary.cpu_model_from_run_dir(run_dir)
    summary = format_cpu_ops_results.build_summary(
        csvs, metadata, suite_id="cpu/fft", run_dir=run_dir, cpu_model=cpu_model
    )
    lines = run_header(
        title="CPU FFT Benchmark Results",
        suite_id="cpu/fft",
        target=target,
        suite_file=metadata.get("suite_file", ""),
        metadata=metadata,
        run_dir=run_dir,
        run_metadata_name=relative_artifact(run_dir, metadata_path) if metadata_path else None,
        threads=threads,
        command="run_cpu_fft.sh",
        intro=[
            f"This file is generated from sequential CPU FFT runs under `{run_label(run_dir)}`.",
            "",
        ],
    )
    lines += cpu_info_section(run_dir) + thread_env_section(run_dir)
    lines += ["## Timing Discipline", ""] + list(FFT_TIMING_DISCIPLINE) + [""]
    lines += [f"## Threads: {' '.join(str(thread) for thread in threads) or 'unknown'}", ""]
    for csv_path in csvs:
        lines.append(f"- CSV: `{relative_artifact(run_dir, csv_path)}`")
    lines += ["", table.rstrip(), ""]
    return write_outputs(run_dir, out_dir, "report.md", "fft", summary, lines)


def render_permutation(run_dir: Path, out_dir: Path) -> list[Path]:
    import format_permutation_results
    from validate_benchmark_suite import PERMUTATION_RESULT_SCHEMA, validate_results

    inputs = sorted(Path(run_dir).glob("*_output_t*.jsonl"))
    if not inputs:
        raise RenderError(f"{run_dir}: no permutation JSONL outputs")
    for path in inputs:
        if not validate_results(path, PERMUTATION_RESULT_SCHEMA):
            raise RenderError(f"{path}: permutation result schema mismatch")
    metadata, _ = load_metadata(run_dir)
    records = [record for path in inputs for record in format_permutation_results.load_records(path)]
    if not records:
        raise RenderError(f"{run_dir}: no permutation records")
    target = metadata.get("target_profile") or "unknown"
    text = format_permutation_results.format_markdown(records, metadata, run_dir=run_dir)
    summary = format_permutation_results.build_summary(
        records,
        metadata,
        cpu_model=report_summary.cpu_model_from_run_dir(run_dir),
        run_dir=run_dir,
    )
    return write_text_outputs(
        out_dir, {"report.md": text, "permutation.summary.json": summary}
    )


def render_overhead_smoke(run_dir: Path, out_dir: Path) -> list[Path]:
    import benchmark_overhead_smoke

    if not list(Path(run_dir).glob("samples_t*.jsonl")):
        raise RenderError(f"{run_dir}: no overhead smoke samples")
    metadata, _ = load_metadata(run_dir)
    target = metadata.get("target_profile") or "unknown"
    text, summary = benchmark_overhead_smoke.render_report(run_dir, target)
    return write_text_outputs(
        out_dir,
        {
            "report.md": text,
            "overhead_smoke.summary.json": summary,
            "report.summary.json": summary,
        },
    )


def render_small_work(run_dir: Path, out_dir: Path) -> list[Path]:
    import benchmark_small_work

    if not list(Path(run_dir).glob("samples_t*.jsonl")):
        raise RenderError(f"{run_dir}: no small-work samples")
    metadata, _ = load_metadata(run_dir)
    text = benchmark_small_work.render_report(run_dir, metadata.get("target_profile") or "unknown")
    summary = benchmark_small_work.build_summary(
        run_dir, cpu_model=report_summary.cpu_model_from_run_dir(run_dir)
    )
    return write_text_outputs(out_dir, {"report.md": text, "small_work.summary.json": summary})


def render_session_matrix(run_dir: Path, out_dir: Path) -> list[Path]:
    import benchmark_cpu_session

    if not list(Path(run_dir).glob("samples_t*.jsonl")):
        raise RenderError(f"{run_dir}: no session-matrix samples")
    metadata, _ = load_metadata(run_dir)
    text = benchmark_cpu_session.render_report(run_dir, metadata.get("target_profile") or "unknown")
    summary = benchmark_cpu_session.build_summary(
        run_dir, cpu_model=report_summary.cpu_model_from_run_dir(run_dir)
    )
    return write_text_outputs(
        out_dir, {"report.md": text, "session_matrix.summary.json": summary}
    )


def render_perf_issues(run_dir: Path, out_dir: Path) -> list[Path]:
    import benchmark_perf_issues

    if not list(Path(run_dir).glob("samples_t*.jsonl")):
        raise RenderError(f"{run_dir}: no perf-issues samples")
    metadata, _ = load_metadata(run_dir)
    target = metadata.get("target_profile") or "unknown"
    text = benchmark_perf_issues.render_report(run_dir, target)
    summary = benchmark_perf_issues.build_summary(run_dir, target)
    return write_text_outputs(
        out_dir, {"report.md": text, "perf_issues.summary.json": summary}
    )


# --------------------------------------------------------------------------
# Dispatch and output
# --------------------------------------------------------------------------


def write_text_outputs(out_dir: Path, outputs: dict[str, Any]) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, value in outputs.items():
        if name.endswith(".json"):
            if not isinstance(value, dict):
                value = json.loads(value)
            path = report_summary.write_document(out_dir / name, value)
        else:
            path = out_dir / name
            path.write_text(value if value.endswith("\n") else value + "\n")
        written.append(path)
    return written


def write_outputs(
    run_dir: Path,
    out_dir: Path,
    report_name: str,
    suite_name: str,
    summary: dict[str, Any],
    lines: list[str],
) -> list[Path]:
    return write_text_outputs(
        out_dir,
        {
            report_name: "\n".join(lines).rstrip() + "\n",
            f"{suite_name}.summary.json": summary,
        },
    )


RENDERERS = {
    "cpu/einsum": render_einsum,
    "cpu/public_api": render_public_api,
    "cpu/permutation": render_permutation,
    "cpu/fft": render_fft,
    "cpu/overhead_smoke": render_overhead_smoke,
    "cpu/small_work": render_small_work,
    "cpu/session_matrix": render_session_matrix,
    "cpu/perf_issues": render_perf_issues,
}

# Suites whose run directory holds more than one report.
EXTRA_RENDERERS = {
    "cpu/einsum": (render_cpu_ops, render_linalg_jvp_vjp),
}


def detect_suite(run_dir: Path, metadata: dict[str, Any]) -> str:
    suite_id = metadata.get("suite_id") or ""
    if suite_id in RENDERERS:
        return suite_id
    if suite_id.startswith("cpu/public_api"):
        return "cpu/public_api"
    # A run directory is data/results/<profile>/cpu/<suite>/<timestamp>.
    parts = Path(run_dir).resolve().parts
    if len(parts) >= 2 and parts[-3] == "cpu":
        candidate = f"cpu/{parts[-2]}"
        if candidate in RENDERERS:
            return candidate
        if candidate.startswith("cpu/public_api"):
            return "cpu/public_api"
    raise RenderError(f"{run_dir}: cannot determine the suite to render")


def render(run_dir: Path, out_dir: Path | None = None) -> list[Path]:
    """Render every report a run directory can produce; deterministic in its inputs."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise RenderError(f"{run_dir}: not a directory")
    out_dir = Path(out_dir).resolve() if out_dir is not None else run_dir
    metadata, _ = load_metadata(run_dir)
    suite_id = detect_suite(run_dir, metadata)
    written = RENDERERS[suite_id](run_dir, out_dir)
    for extra in EXTRA_RENDERERS.get(suite_id, ()):
        written.extend(extra(run_dir, out_dir))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: run dir)")
    args = parser.parse_args(argv)
    try:
        written = render(args.run_dir, args.out)
    except RenderError as exc:
        print(f"render_run: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
