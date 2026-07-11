#!/usr/bin/env python3
"""Generate benchmarks/gpu/linalg_jvp_vjp.yaml from CPU publication-gate seeds."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "gpu" / "linalg_jvp_vjp.yaml"

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


def build_problems() -> list[dict]:
    problems: list[dict] = []
    for suite, sizes in (("small", SMALL_SIZES), ("large", LARGE_SIZES)):
        for n in sizes:
            for loss in LOSSES:
                seeds = SEEDS[suite][loss]
                rhs_cols = seeds.get("rhs_cols")
                shape = shape_label(loss, n, rhs_cols)
                linalg_ad = {
                    "suite": suite,
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
                            "id": f"linalg_ad_{suite}_{loss}_{phase}_f64_{n}",
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


def main() -> None:
    suite = {
        "schema_version": 1,
        "suite_id": "gpu/linalg_jvp_vjp",
        "title": "GPU Linalg JVP/VJP Benchmarks",
        "description": (
            "Trace-mode linalg automatic differentiation on CUDA for tenferro-rs, "
            "PyTorch, and JAX."
        ),
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
            "jax-cuda",
        ],
        "problems": build_problems(),
    }
    OUT.write_text(yaml.safe_dump(suite, sort_keys=False))
    print(f"Wrote {OUT} ({len(suite['problems'])} problems)")


if __name__ == "__main__":
    main()
