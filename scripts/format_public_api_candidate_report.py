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
    report.write_text(header + notes + format_table(csvs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    refresh(parser.parse_args().report)
