#!/usr/bin/env python3
"""Generate benchmarks/gpu/linalg_jvp_vjp.yaml from CPU publication-gate seeds."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

SMALL_SIZES = [2, 4, 8]
LARGE_SIZES = [256, 512]

SEEDS = {
    "small": {
        "grad_sum_svd_s": {"matrix_seed": 10},
        "grad_sum_qr": {"matrix_seed": 4},
        "grad_sum_eigh": {"matrix_seed": 5},
        "grad_sum_lu": {"matrix_seed": 52},
        "grad_sum_solve": {"matrix_seed": 11, "rhs_seed": 12, "rhs_cols": 1},
    },
    "large": {
        "grad_sum_svd_s": {"matrix_seed": 34},
        "grad_sum_qr": {"matrix_seed": 26},
        "grad_sum_eigh": {"matrix_seed": 27},
        "grad_sum_lu": {"matrix_seed": 54},
        "grad_sum_solve": {"matrix_seed": 35, "rhs_seed": 36, "rhs_cols": 1},
    },
}

LOSSES = [
    "grad_sum_svd_s",
    "grad_sum_qr",
    "grad_sum_eigh",
    "grad_sum_lu",
    "grad_sum_solve",
]


def shape_label(loss: str, n: int, rhs_cols: int | None) -> str:
    if loss == "grad_sum_solve":
        return f"{n}x{n},rhs={rhs_cols}"
    return f"{n}x{n}"


def generator_for(loss: str) -> str:
    if loss in {"grad_sum_eigh", "grad_sum_solve"}:
        return "spd"
    return "well_conditioned"


def build_problems(regime: str) -> list[dict]:
    problems: list[dict] = []
    sizes = SMALL_SIZES if regime == "small" else LARGE_SIZES
    for n in sizes:
        for loss in LOSSES:
            seeds = SEEDS[regime][loss]
            rhs_cols = seeds.get("rhs_cols")
            shape = shape_label(loss, n, rhs_cols)
            linalg_ad = {
                "suite": regime,
                "loss": loss,
                "n": n,
                "matrix_seed": seeds["matrix_seed"],
            }
            if loss == "grad_sum_solve":
                linalg_ad["rhs_seed"] = seeds["rhs_seed"]
                linalg_ad["rhs_cols"] = rhs_cols
            for phase in ("jvp", "vjp"):
                problems.append(
                    {
                        "id": f"linalg_ad_{regime}_{loss}_{phase}_f64_{n}",
                        "family": "linalg_ad",
                        "op": phase,
                        "dtype": {"a": "f64"},
                        "layout": {
                            "tenferro": "col_major",
                            "framework": "row_major_contiguous",
                        },
                        "data": {
                            "generator": generator_for(loss),
                            "seed": seeds["matrix_seed"],
                        },
                        "linalg_ad": linalg_ad,
                        "run": {
                            "warmups": 3,
                            "runs": 7,
                            "timing_scope": "steady_state_host_api_plus_device_sync",
                        },
                        "verify": {
                            "reference": "cpu_fp64",
                            "rtol": 1.0e-5,
                            "atol": 1.0e-8,
                        },
                        "notes": f"shape={shape}",
                    }
                )
    return problems


def suite_document(suite_id: str, title: str, description: str, regime: str) -> dict:
    return {
        "schema_version": 1,
        "suite_id": suite_id,
        "title": title,
        "description": description,
        "defaults": {
            "device": {"kind": "cuda", "ordinal": 0},
            "layout": {
                "tenferro": "col_major",
                "framework": "row_major_contiguous",
            },
            "run": {
                "warmups": 3,
                "runs": 7,
                "timing_scope": "steady_state_host_api_plus_device_sync",
            },
            "verify": {"reference": "cpu_fp64", "rtol": 1.0e-5, "atol": 1.0e-8},
        },
        "backends": [
            "tenferro-cuda-trace",
            "pytorch-cuda",
        ],
        "problems": build_problems(regime),
    }


# `gpu/linalg_jvp_vjp` keeps the GPU-sized rows. The n=2/4/8 rows measure
# single-call latency and per-op overhead (the device kernels are a small
# fraction of the wall time), so they live in `gpu/linalg_ad_latency` and must
# not be read as GPU throughput numbers.
SUITES = {
    "linalg_jvp_vjp": suite_document(
        "gpu/linalg_jvp_vjp",
        "GPU Linalg JVP/VJP Benchmarks",
        (
            "Trace-mode linalg automatic differentiation on CUDA for tenferro-rs "
            "and PyTorch at GPU-sized matrices (n=256, 512)."
        ),
        "large",
    ),
    "linalg_ad_latency": suite_document(
        "gpu/linalg_ad_latency",
        "GPU Linalg AD Single-Call Latency Benchmarks",
        (
            "Single-call latency and per-op overhead of trace-mode linalg AD on CUDA "
            "at n=2, 4, 8. Device kernel time is a small fraction of each measurement, "
            "so these rows are latency/overhead diagnostics, not GPU throughput."
        ),
        "small",
    ),
}


def main() -> None:
    for name, suite in SUITES.items():
        out = ROOT / "benchmarks" / "gpu" / f"{name}.yaml"
        out.write_text(yaml.safe_dump(suite, sort_keys=False))
        print(f"Wrote {out} ({len(suite['problems'])} problems)")


if __name__ == "__main__":
    main()
