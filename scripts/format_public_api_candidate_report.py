#!/usr/bin/env python3
"""Regenerate a focused report from recorded metadata and unchanged raw CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path

from format_cpu_ops_results import format_table


def refresh(report: Path) -> None:
    # Preserve the recorded revision, timestamp, CPU and thread environments.
    header = report.read_text().split("## Timing Discipline", 1)[0]
    csvs = sorted(report.parent.glob("cpu_public_api_t*.csv"))
    notes = """## Timing Discipline

- Fixtures, Tensor/EagerTensor construction, session entry, input copies, destinations, library initialization and JAX compilation are completed explicitly before sampling, including zero-warmup runs.
- Direct and eager paths reuse one entered session. Outputs remain alive until the clock stops; numerical validation and destruction occur outside timing.
- Compact Householder QR measures compact-state initialization. Q/R extraction, reconstruction and orthogonality checks are untimed.
- RRQR requests Q, R, column permutation and numerical rank, with Raw gauge, rtol=1e-12 and atol=0. Full-rank and deficient fixtures are separate cases. PyTorch has no pivoted QR; JAX's three-output pivoted QR is not a matching four-output operation.
- Caller-output solve preallocates its destination and includes factorization plus solve. PyTorch uses torch.linalg.solve(out=...). JAX and eager/trace caller-output solve are unsupported.
- Masked log-softmax compares the public tenferro operation with jax.nn.log_softmax(where=mask). PyTorch's mask-plus-log-softmax composition is excluded. Axis-wise sum-of-squares compositions are also excluded.
- Selection compares torch.where/jnp.where with identical logical values in each framework's native layout. The existing select row covers the same backend kernel; this row explicitly measures the public session spelling.
- Candidate operations use batches of sixteen retained outputs, normalized per operation. Selection is memory-limited to one operation because each output is 256 MiB. Batch counts and durations are preserved in the CSV notes and table below.
- Compiled trace execution has no public borrowed BackendSession API. Its per-call session initialization is excluded from these standard operation rows; compact-state trace also cannot return the opaque state without extra operations.
- Provider and linked-library observations are recorded in run_t1.yaml and run_t4.yaml. The CPU model above is the actual Docker host; the amd-cpu path is the repository's standard Linux CPU collection profile.

"""
    if report.parent.parent.name == "public_api_followups":
        notes = """## Timing Discipline

- These thirteen APIs already have AMD perf_issues measurements; this collection adds dedicated standard public-operation cases and JAX comparisons rather than claiming they were previously unmeasured.
- All cases use deterministic F32 logical matrices in native framework layouts. Direct and eager reuse entered sessions; inputs, masks, indices, runtime construction, initialization, JIT preparation and numerical validation are outside timing, including zero warmups.
- The small 1024x64 cases batch 64 operations and the 4096x256 control cases batch eight, retaining every output through clock stop. Batch duration and normalized per-operation timing are recorded.
- Axis operations reduce or normalize axis 1. LayerNorm and RMSNorm are non-affine with eps=1e-5; affine variants are not claimed as covered. JAX standardize(algorithm=stable) provides LayerNorm, and standardize(mean=0, algorithm=stable) provides non-affine RMSNorm directly.
- gelu requests the exact erf formulation; gelu_tanh requests the tanh approximation. Softplus inputs stay below the PyTorch threshold transition. Masked softmax requests zero masked outputs; JAX uses its where argument and the missing direct PyTorch contract is unsupported.
- Each route is validated against independent double-precision scalar/NumPy references. Operation-created outputs are timed; validation copies and output destruction are untimed. Eager capture is inactive and inputs are untracked.
- Compiled trace has no public borrowed-session execution API, so its session creation is excluded through explicit unsupported rows. Julia columns are unselected.
- Provider/configuration/linked-library and thread observations are recorded in run_t1.yaml and run_t4.yaml. The CPU model above identifies this Docker host; these results are not a speedup comparison to historical AMD hardware.

"""
    report.write_text(header + notes + format_table(csvs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    refresh(parser.parse_args().report)
