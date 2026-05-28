# GPU Backend Comparison Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the GPU benchmark suite contract: suite schema, result schema, sample suites, validation, JSONL formatting, and a non-GPU orchestration smoke path.

**Architecture:** Add the GPU benchmark layer beside the current CPU runners. Phase 1 does not measure GPU kernels; it establishes stable file formats and report generation so measured tenferro/PyTorch/LibTorch/JAX/vendor runners can plug in through separate follow-up implementation plans without changing the contract.

**Tech Stack:** JSON Schema draft 2020-12, YAML via PyYAML, Python stdlib JSON/argparse, Bash, existing `data/results/` and `result/` report layout.

---

## Scope

This plan implements Phase 1 from `docs/superpowers/specs/2026-05-28-gpu-backend-comparison-design.md`.

It intentionally does not implement measured CUDA runners. Every backend row in the smoke path is produced from explicit JSONL fixture records or structured `not_configured` records. That keeps this work testable on machines without an NVIDIA GPU.

## File Structure

- Create `schemas/benchmark-suite.schema.json`: validates GPU benchmark suite YAML after YAML is loaded as JSON-compatible data.
- Create `schemas/benchmark-result.schema.json`: validates one JSONL result record.
- Create `benchmarks/gpu/dense.yaml`: representative dense problems for `matmul`, `batched_matmul`, `qr`, `solve`, `svd`, and `eigh`.
- Create `benchmarks/gpu/einsum.yaml`: representative `einsum` problem adapted from current `data/instances/bin_matmul_256.json`.
- Create `benchmarks/gpu/sparse.yaml`: representative `spmv` and `spmm` sparse problems.
- Create `scripts/validate_benchmark_suite.py`: validates suite files and JSONL result files.
- Create `scripts/format_gpu_results.py`: reads JSONL records and writes a Markdown comparison table.
- Create `scripts/run_gpu_suite.sh`: validates suites, emits structured `not_configured` records for requested backends in Phase 1, and formats reports.
- Create `tests/test_gpu_benchmark_contract.sh`: shell test that exercises validation, formatting, and the Phase 1 smoke orchestration.
- Modify `README.md`: add a short GPU contract workflow section.

## Task 1: Add Suite Schema

**Files:**
- Create: `schemas/benchmark-suite.schema.json`

- [ ] **Step 1: Create the failing schema existence test**

Run:

```bash
test -f schemas/benchmark-suite.schema.json
```

Expected: FAIL because the file does not exist.

- [ ] **Step 2: Add the suite schema**

Create `schemas/benchmark-suite.schema.json` with this exact content:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://tensor4all.example/schemas/benchmark-suite.schema.json",
  "title": "GPU Benchmark Suite",
  "type": "object",
  "required": ["schema_version", "suite_id", "description", "problems"],
  "additionalProperties": false,
  "properties": {
    "schema_version": { "const": 1 },
    "suite_id": { "type": "string", "minLength": 1 },
    "description": { "type": "string", "minLength": 1 },
    "defaults": {
      "type": "object",
      "additionalProperties": true,
      "properties": {
        "run": { "$ref": "#/$defs/run" },
        "verify": { "$ref": "#/$defs/verify" },
        "device": { "$ref": "#/$defs/device" },
        "layout": { "$ref": "#/$defs/layout" }
      }
    },
    "problems": {
      "type": "array",
      "minItems": 1,
      "items": { "$ref": "#/$defs/problem" }
    }
  },
  "$defs": {
    "device": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "kind": { "enum": ["cuda"] },
        "ordinal": { "type": "integer", "minimum": 0 }
      }
    },
    "run": {
      "type": "object",
      "required": ["warmups", "runs"],
      "additionalProperties": false,
      "properties": {
        "warmups": { "type": "integer", "minimum": 0 },
        "runs": { "type": "integer", "minimum": 1 },
        "min_runtime_ms": { "type": "number", "minimum": 0 },
        "timing_scope": {
          "enum": [
            "steady_state_device_only",
            "steady_state_host_api_plus_device_sync",
            "steady_state_host_api_plus_device_sync_with_download",
            "compile_plus_first_run"
          ]
        }
      }
    },
    "verify": {
      "type": "object",
      "required": ["reference", "rtol", "atol"],
      "additionalProperties": true,
      "properties": {
        "reference": { "type": "string", "minLength": 1 },
        "rtol": { "type": "number", "minimum": 0 },
        "atol": { "type": "number", "minimum": 0 },
        "residual_rtol": { "type": "number", "minimum": 0 }
      }
    },
    "layout": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "tenferro": { "enum": ["col_major"] },
        "framework": { "enum": ["row_major_contiguous"] },
        "sparse": { "enum": ["csr"] }
      }
    },
    "dtype": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "a": { "type": "string" },
        "b": { "type": "string" },
        "c": { "type": "string" },
        "values": { "type": "string" },
        "x": { "type": "string" },
        "y": { "type": "string" },
        "accum": { "type": "string" }
      }
    },
    "data": {
      "type": "object",
      "required": ["generator", "seed"],
      "additionalProperties": true,
      "properties": {
        "generator": {
          "enum": [
            "normal",
            "zeros",
            "well_conditioned",
            "spd",
            "symmetric_normal",
            "suitesparse"
          ]
        },
        "seed": { "type": "integer" }
      }
    },
    "problem": {
      "type": "object",
      "required": [
        "id",
        "family",
        "op",
        "dtype",
        "data",
        "run",
        "verify",
        "backend_candidates"
      ],
      "additionalProperties": false,
      "properties": {
        "id": { "type": "string", "minLength": 1 },
        "family": { "enum": ["dense", "sparse", "einsum"] },
        "op": {
          "enum": [
            "matmul",
            "batched_matmul",
            "einsum",
            "qr",
            "solve",
            "svd",
            "eigh",
            "spmv",
            "spmm"
          ]
        },
        "dtype": { "$ref": "#/$defs/dtype" },
        "layout": { "$ref": "#/$defs/layout" },
        "data": { "$ref": "#/$defs/data" },
        "run": { "$ref": "#/$defs/run" },
        "verify": { "$ref": "#/$defs/verify" },
        "backend_candidates": {
          "type": "array",
          "minItems": 1,
          "items": { "type": "string", "minLength": 1 }
        },
        "matmul": { "$ref": "#/$defs/matmul" },
        "batched_matmul": { "$ref": "#/$defs/batched_matmul" },
        "einsum": { "$ref": "#/$defs/einsum" },
        "linalg": { "$ref": "#/$defs/linalg" },
        "sparse": { "$ref": "#/$defs/sparse" }
      },
      "allOf": [
        {
          "if": { "properties": { "op": { "const": "matmul" } } },
          "then": { "required": ["matmul"] }
        },
        {
          "if": { "properties": { "op": { "const": "batched_matmul" } } },
          "then": { "required": ["batched_matmul"] }
        },
        {
          "if": { "properties": { "op": { "const": "einsum" } } },
          "then": { "required": ["einsum"] }
        },
        {
          "if": {
            "properties": {
              "op": { "enum": ["qr", "solve", "svd", "eigh"] }
            }
          },
          "then": { "required": ["linalg"] }
        },
        {
          "if": { "properties": { "op": { "enum": ["spmv", "spmm"] } } },
          "then": { "required": ["sparse"] }
        }
      ]
    },
    "matmul": {
      "type": "object",
      "required": ["m", "n", "k"],
      "additionalProperties": false,
      "properties": {
        "m": { "type": "integer", "minimum": 1 },
        "n": { "type": "integer", "minimum": 1 },
        "k": { "type": "integer", "minimum": 1 },
        "transpose_a": { "type": "boolean" },
        "transpose_b": { "type": "boolean" }
      }
    },
    "batched_matmul": {
      "type": "object",
      "required": ["batch", "m", "n", "k", "batch_layout"],
      "additionalProperties": false,
      "properties": {
        "batch": { "type": "integer", "minimum": 1 },
        "m": { "type": "integer", "minimum": 1 },
        "n": { "type": "integer", "minimum": 1 },
        "k": { "type": "integer", "minimum": 1 },
        "batch_layout": { "enum": ["leading", "trailing"] }
      }
    },
    "einsum": {
      "type": "object",
      "required": [
        "format_rowmajor",
        "format_colmajor",
        "shapes_rowmajor",
        "shapes_colmajor",
        "path"
      ],
      "additionalProperties": false,
      "properties": {
        "format_rowmajor": { "type": "string", "minLength": 1 },
        "format_colmajor": { "type": "string", "minLength": 1 },
        "shapes_rowmajor": { "$ref": "#/$defs/shape_list" },
        "shapes_colmajor": { "$ref": "#/$defs/shape_list" },
        "path": {
          "type": "object",
          "required": ["strategy", "pairs"],
          "additionalProperties": false,
          "properties": {
            "strategy": { "enum": ["opt_flops", "opt_size", "manual"] },
            "pairs": {
              "type": "array",
              "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": { "type": "integer", "minimum": 0 }
              }
            }
          }
        }
      }
    },
    "shape_list": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "array",
        "minItems": 1,
        "items": { "type": "integer", "minimum": 1 }
      }
    },
    "linalg": {
      "type": "object",
      "required": ["n"],
      "additionalProperties": false,
      "properties": {
        "m": { "type": "integer", "minimum": 1 },
        "n": { "type": "integer", "minimum": 1 },
        "rhs_cols": { "type": "integer", "minimum": 1 },
        "batch": { "type": "integer", "minimum": 1 },
        "full_matrices": { "type": "boolean" }
      }
    },
    "sparse": {
      "type": "object",
      "required": ["source", "storage", "rows", "cols", "nnz"],
      "additionalProperties": false,
      "properties": {
        "source": { "enum": ["suitesparse", "matrix_market"] },
        "group": { "type": "string" },
        "name": { "type": "string" },
        "storage": { "enum": ["csr"] },
        "rows": { "type": "integer", "minimum": 1 },
        "cols": { "type": "integer", "minimum": 1 },
        "nnz": { "type": "integer", "minimum": 0 },
        "rhs_cols": { "type": "integer", "minimum": 1 }
      }
    }
  }
}
```

- [ ] **Step 3: Verify the file exists**

Run:

```bash
test -f schemas/benchmark-suite.schema.json
```

Expected: PASS.

- [ ] **Step 4: Commit**

Run:

```bash
git add schemas/benchmark-suite.schema.json
git commit -m "feat: add GPU benchmark suite schema"
```

## Task 2: Add Result Schema

**Files:**
- Create: `schemas/benchmark-result.schema.json`

- [ ] **Step 1: Create the failing schema existence test**

Run:

```bash
test -f schemas/benchmark-result.schema.json
```

Expected: FAIL because the file does not exist.

- [ ] **Step 2: Add the result schema**

Create `schemas/benchmark-result.schema.json` with this exact content:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://tensor4all.example/schemas/benchmark-result.schema.json",
  "title": "GPU Benchmark Result Record",
  "type": "object",
  "required": [
    "schema_version",
    "suite_id",
    "problem_id",
    "op",
    "backend",
    "status",
    "timing",
    "verification",
    "environment",
    "execution"
  ],
  "additionalProperties": false,
  "properties": {
    "schema_version": { "const": 1 },
    "suite_id": { "type": "string", "minLength": 1 },
    "problem_id": { "type": "string", "minLength": 1 },
    "op": {
      "enum": [
        "matmul",
        "batched_matmul",
        "einsum",
        "qr",
        "solve",
        "svd",
        "eigh",
        "spmv",
        "spmm"
      ]
    },
    "backend": { "type": "string", "minLength": 1 },
    "status": {
      "enum": [
        "ok",
        "unsupported",
        "not_configured",
        "compile_failed",
        "runtime_failed",
        "oom",
        "verification_failed",
        "cpu_fallback"
      ]
    },
    "timing": {
      "type": "object",
      "required": [
        "warmup_runs",
        "timed_runs",
        "compile_time_ms",
        "first_run_ms",
        "median_ms",
        "min_ms",
        "p95_ms",
        "iqr_ms",
        "timing_scope"
      ],
      "additionalProperties": false,
      "properties": {
        "warmup_runs": { "type": "integer", "minimum": 0 },
        "timed_runs": { "type": "integer", "minimum": 0 },
        "compile_time_ms": { "type": ["number", "null"], "minimum": 0 },
        "first_run_ms": { "type": ["number", "null"], "minimum": 0 },
        "median_ms": { "type": ["number", "null"], "minimum": 0 },
        "min_ms": { "type": ["number", "null"], "minimum": 0 },
        "p95_ms": { "type": ["number", "null"], "minimum": 0 },
        "iqr_ms": { "type": ["number", "null"], "minimum": 0 },
        "timing_scope": {
          "enum": [
            "steady_state_device_only",
            "steady_state_host_api_plus_device_sync",
            "steady_state_host_api_plus_device_sync_with_download",
            "compile_plus_first_run"
          ]
        }
      }
    },
    "performance": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "tflops": { "type": ["number", "null"], "minimum": 0 },
        "effective_bandwidth_gbps": { "type": ["number", "null"], "minimum": 0 },
        "peak_memory_bytes": { "type": ["integer", "null"], "minimum": 0 }
      }
    },
    "verification": {
      "type": "object",
      "required": ["status", "reference_backend", "rtol", "atol"],
      "additionalProperties": false,
      "properties": {
        "status": { "enum": ["passed", "failed", "skipped", "not_applicable"] },
        "reference_backend": { "type": ["string", "null"] },
        "max_abs_error": { "type": ["number", "null"] },
        "max_rel_error": { "type": ["number", "null"] },
        "residual": { "type": ["number", "null"] },
        "rtol": { "type": ["number", "null"], "minimum": 0 },
        "atol": { "type": ["number", "null"], "minimum": 0 },
        "reason": { "type": ["string", "null"] }
      }
    },
    "environment": {
      "type": "object",
      "required": ["timestamp_utc"],
      "additionalProperties": true,
      "properties": {
        "hostname": { "type": ["string", "null"] },
        "timestamp_utc": { "type": "string", "minLength": 1 },
        "os": { "type": ["string", "null"] },
        "gpu_name": { "type": ["string", "null"] },
        "gpu_uuid": { "type": ["string", "null"] },
        "gpu_memory_bytes": { "type": ["integer", "null"], "minimum": 0 },
        "driver_version": { "type": ["string", "null"] },
        "cuda_version": { "type": ["string", "null"] },
        "cudnn_version": { "type": ["string", "null"] },
        "framework_version": { "type": ["string", "null"] },
        "tenferro_rs_commit": { "type": ["string", "null"] },
        "benchmark_repo_commit": { "type": ["string", "null"] },
        "env": {
          "type": "object",
          "additionalProperties": { "type": ["string", "null"] }
        }
      }
    },
    "execution": {
      "type": "object",
      "required": [
        "device",
        "device_ordinal",
        "execution_path",
        "synchronization",
        "layout",
        "dtype"
      ],
      "additionalProperties": false,
      "properties": {
        "device": { "enum": ["cuda"] },
        "device_ordinal": { "type": "integer", "minimum": 0 },
        "execution_path": { "type": "string" },
        "synchronization": { "type": "string" },
        "layout": { "type": "string" },
        "dtype": { "type": "string" },
        "notes": { "type": ["string", "null"] },
        "unsupported_reason": { "type": ["string", "null"] }
      }
    }
  }
}
```

- [ ] **Step 3: Verify the file exists**

Run:

```bash
test -f schemas/benchmark-result.schema.json
```

Expected: PASS.

- [ ] **Step 4: Commit**

Run:

```bash
git add schemas/benchmark-result.schema.json
git commit -m "feat: add GPU benchmark result schema"
```

## Task 3: Add Sample GPU Suites

**Files:**
- Create: `benchmarks/gpu/dense.yaml`
- Create: `benchmarks/gpu/einsum.yaml`
- Create: `benchmarks/gpu/sparse.yaml`

- [ ] **Step 1: Create failing existence test**

Run:

```bash
test -f benchmarks/gpu/dense.yaml
test -f benchmarks/gpu/einsum.yaml
test -f benchmarks/gpu/sparse.yaml
```

Expected: FAIL because the files do not exist.

- [ ] **Step 2: Add `benchmarks/gpu/dense.yaml`**

Create `benchmarks/gpu/dense.yaml`:

```yaml
schema_version: 1
suite_id: gpu_dense_contract_v1
description: Dense GPU benchmark contract suite for matmul, batched matmul, and linalg ops.
defaults:
  device: {kind: cuda, ordinal: 0}
  layout: {tenferro: col_major, framework: row_major_contiguous}
  run: {warmups: 3, runs: 7, timing_scope: steady_state_host_api_plus_device_sync}
  verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10}
problems:
  - id: dense_matmul_f64_256
    family: dense
    op: matmul
    dtype: {a: f64, b: f64, c: f64, accum: f64}
    layout: {tenferro: col_major, framework: row_major_contiguous}
    data: {generator: normal, seed: 1}
    matmul: {m: 256, n: 256, k: 256, transpose_a: false, transpose_b: false}
    run: {warmups: 3, runs: 7, timing_scope: steady_state_host_api_plus_device_sync}
    verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10}
    backend_candidates: [tenferro-cuda-trace, tenferro-cuda-eager, pytorch-cuda, libtorch-cuda, jax-cuda, cublaslt, cutlass]

  - id: dense_batched_matmul_f64_b16_32
    family: dense
    op: batched_matmul
    dtype: {a: f64, b: f64, c: f64, accum: f64}
    layout: {tenferro: col_major, framework: row_major_contiguous}
    data: {generator: normal, seed: 2}
    batched_matmul: {batch: 16, m: 32, n: 32, k: 32, batch_layout: leading}
    run: {warmups: 3, runs: 7, timing_scope: steady_state_host_api_plus_device_sync}
    verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10}
    backend_candidates: [tenferro-cuda-trace, tenferro-cuda-eager, pytorch-cuda, libtorch-cuda, jax-cuda, cublaslt, cutlass]

  - id: dense_qr_f64_64
    family: dense
    op: qr
    dtype: {a: f64}
    layout: {tenferro: col_major, framework: row_major_contiguous}
    data: {generator: well_conditioned, seed: 3}
    linalg: {m: 64, n: 64, full_matrices: false}
    run: {warmups: 2, runs: 5, timing_scope: steady_state_host_api_plus_device_sync}
    verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10}
    backend_candidates: [tenferro-cuda-trace, tenferro-cuda-eager, pytorch-cuda, libtorch-cuda, jax-cuda, cusolver]

  - id: dense_solve_f64_64_rhs4
    family: dense
    op: solve
    dtype: {a: f64, b: f64, c: f64}
    layout: {tenferro: col_major, framework: row_major_contiguous}
    data: {generator: spd, seed: 4}
    linalg: {n: 64, rhs_cols: 4}
    run: {warmups: 2, runs: 5, timing_scope: steady_state_host_api_plus_device_sync}
    verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10, residual_rtol: 1.0e-8}
    backend_candidates: [tenferro-cuda-trace, tenferro-cuda-eager, pytorch-cuda, libtorch-cuda, jax-cuda, cusolver]

  - id: dense_svd_f64_64
    family: dense
    op: svd
    dtype: {a: f64}
    layout: {tenferro: col_major, framework: row_major_contiguous}
    data: {generator: well_conditioned, seed: 5}
    linalg: {m: 64, n: 64, full_matrices: false}
    run: {warmups: 2, runs: 5, timing_scope: steady_state_host_api_plus_device_sync}
    verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10}
    backend_candidates: [tenferro-cuda-trace, tenferro-cuda-eager, pytorch-cuda, libtorch-cuda, jax-cuda, cusolver]

  - id: dense_eigh_f64_64
    family: dense
    op: eigh
    dtype: {a: f64}
    layout: {tenferro: col_major, framework: row_major_contiguous}
    data: {generator: spd, seed: 6}
    linalg: {n: 64}
    run: {warmups: 2, runs: 5, timing_scope: steady_state_host_api_plus_device_sync}
    verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10, residual_rtol: 1.0e-8}
    backend_candidates: [tenferro-cuda-trace, tenferro-cuda-eager, pytorch-cuda, libtorch-cuda, jax-cuda, cusolver]
```

- [ ] **Step 3: Add `benchmarks/gpu/einsum.yaml`**

Create `benchmarks/gpu/einsum.yaml`:

```yaml
schema_version: 1
suite_id: gpu_einsum_contract_v1
description: GPU einsum benchmark contract suite adapted from existing benchmark instances.
defaults:
  device: {kind: cuda, ordinal: 0}
  layout: {tenferro: col_major, framework: row_major_contiguous}
  run: {warmups: 3, runs: 7, timing_scope: steady_state_host_api_plus_device_sync}
  verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10}
problems:
  - id: einsum_bin_matmul_256_f64
    family: einsum
    op: einsum
    dtype: {a: f64, b: f64, c: f64, accum: f64}
    layout: {tenferro: col_major, framework: row_major_contiguous}
    data: {generator: zeros, seed: 0}
    einsum:
      format_rowmajor: "ij,jk->ik"
      format_colmajor: "ij,jk->ik"
      shapes_rowmajor: [[256, 256], [256, 256]]
      shapes_colmajor: [[256, 256], [256, 256]]
      path: {strategy: opt_flops, pairs: [[0, 1]]}
    run: {warmups: 3, runs: 7, timing_scope: steady_state_host_api_plus_device_sync}
    verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10}
    backend_candidates: [tenferro-cuda-trace, tenferro-cuda-eager, pytorch-cuda, libtorch-cuda, jax-cuda, cublaslt, cutlass]
```

- [ ] **Step 4: Add `benchmarks/gpu/sparse.yaml`**

Create `benchmarks/gpu/sparse.yaml`:

```yaml
schema_version: 1
suite_id: gpu_sparse_contract_v1
description: Sparse GPU benchmark contract suite using SuiteSparse-style metadata.
defaults:
  device: {kind: cuda, ordinal: 0}
  layout: {sparse: csr}
  run: {warmups: 3, runs: 7, timing_scope: steady_state_host_api_plus_device_sync}
  verify: {reference: cpu_sparse_fp64, rtol: 1.0e-8, atol: 1.0e-10}
problems:
  - id: sparse_bcspwr10_spmv_f64
    family: sparse
    op: spmv
    dtype: {values: f64, x: f64, y: f64}
    layout: {sparse: csr}
    data: {generator: suitesparse, seed: 0}
    sparse: {source: suitesparse, group: HB, name: bcspwr10, storage: csr, rows: 5300, cols: 5300, nnz: 21842}
    run: {warmups: 3, runs: 7, timing_scope: steady_state_host_api_plus_device_sync}
    verify: {reference: cpu_sparse_fp64, rtol: 1.0e-8, atol: 1.0e-10}
    backend_candidates: [cusparse, ginkgo, pytorch-cuda, libtorch-cuda]

  - id: sparse_bcsstk17_spmm_f64_rhs32
    family: sparse
    op: spmm
    dtype: {values: f64, x: f64, y: f64}
    layout: {sparse: csr}
    data: {generator: suitesparse, seed: 0}
    sparse: {source: suitesparse, group: HB, name: bcsstk17, storage: csr, rows: 10974, cols: 10974, nnz: 428650, rhs_cols: 32}
    run: {warmups: 3, runs: 7, timing_scope: steady_state_host_api_plus_device_sync}
    verify: {reference: cpu_sparse_fp64, rtol: 1.0e-8, atol: 1.0e-10}
    backend_candidates: [cusparse, ginkgo, pytorch-cuda, libtorch-cuda]
```

- [ ] **Step 5: Verify files exist**

Run:

```bash
test -f benchmarks/gpu/dense.yaml
test -f benchmarks/gpu/einsum.yaml
test -f benchmarks/gpu/sparse.yaml
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add benchmarks/gpu/dense.yaml benchmarks/gpu/einsum.yaml benchmarks/gpu/sparse.yaml
git commit -m "feat: add GPU benchmark contract suites"
```

## Task 4: Add Validation Script

**Files:**
- Create: `scripts/validate_benchmark_suite.py`

- [ ] **Step 1: Write failing validation command**

Run:

```bash
python3 scripts/validate_benchmark_suite.py benchmarks/gpu/dense.yaml
```

Expected: FAIL with `No such file or directory` or `can't open file`.

- [ ] **Step 2: Add the validation script**

Create `scripts/validate_benchmark_suite.py`:

```python
#!/usr/bin/env python3
"""Validate GPU benchmark suite YAML files and result JSONL records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

PROJECT_DIR = Path(__file__).resolve().parents[1]
SUITE_SCHEMA = PROJECT_DIR / "schemas" / "benchmark-suite.schema.json"
RESULT_SCHEMA = PROJECT_DIR / "schemas" / "benchmark-result.schema.json"


def load_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def load_yaml(path: Path) -> Any:
    with path.open() as fh:
        return yaml.safe_load(fh)


def validator_for(schema_path: Path) -> Draft202012Validator:
    return Draft202012Validator(load_json(schema_path))


def print_errors(path: Path, errors: list[Any]) -> None:
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path)
        prefix = f"{path}:{location}" if location else str(path)
        print(f"{prefix}: {error.message}", file=sys.stderr)


def validate_object(path: Path, obj: Any, validator: Draft202012Validator) -> bool:
    errors = sorted(validator.iter_errors(obj), key=lambda err: list(err.absolute_path))
    if errors:
        print_errors(path, errors)
        return False
    return True


def validate_suite(path: Path) -> bool:
    return validate_object(path, load_yaml(path), validator_for(SUITE_SCHEMA))


def validate_results(path: Path) -> bool:
    validator = validator_for(RESULT_SCHEMA)
    ok = True
    with path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(f"{path}:{line_no}: invalid JSON: {exc}", file=sys.stderr)
                ok = False
                continue
            if not validate_object(Path(f"{path}:{line_no}"), record, validator):
                ok = False
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--kind",
        choices=["suite", "result"],
        default="suite",
        help="Validate suite YAML or result JSONL records.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ok = True
    for path in args.paths:
        if args.kind == "suite":
            ok = validate_suite(path) and ok
        else:
            ok = validate_results(path) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Ensure Python dependencies are available**

Run:

```bash
uv run python -c 'import yaml, jsonschema; print("ok")'
```

Expected: If this fails, add dependencies in Task 5 before running validation. If it prints `ok`, continue.

- [ ] **Step 4: Run suite validation**

Run:

```bash
uv run python scripts/validate_benchmark_suite.py benchmarks/gpu/dense.yaml benchmarks/gpu/einsum.yaml benchmarks/gpu/sparse.yaml
```

Expected: PASS with no output.

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/validate_benchmark_suite.py
git commit -m "feat: add GPU benchmark validation script"
```

## Task 5: Add Validation Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Check whether `jsonschema` is missing**

Run:

```bash
uv run python -c 'import yaml, jsonschema; print("ok")'
```

Expected: PASS if dependencies already exist transitively; otherwise FAIL with `ModuleNotFoundError`.

- [ ] **Step 2: Add explicit dependency only if Step 1 failed**

Modify `pyproject.toml` dependencies to include `jsonschema>=4.0.0` while keeping existing entries:

```toml
dependencies = [
    "einsum-benchmark>=0.1.7",
    "jax>=0.10.1",
    "jsonschema>=4.0.0",
    "opt-einsum>=3.4.0",
    "torch>=2.12.0",
    "tqdm>=4.67.3",
]
```

- [ ] **Step 3: Sync lockfile if `pyproject.toml` changed**

Run:

```bash
uv lock
```

Expected: PASS and `uv.lock` updated only if a new dependency was needed.

- [ ] **Step 4: Re-run import check**

Run:

```bash
uv run python -c 'import yaml, jsonschema; print("ok")'
```

Expected: prints `ok`.

- [ ] **Step 5: Commit only if files changed**

Run:

```bash
git status --short pyproject.toml uv.lock
git add pyproject.toml uv.lock
git commit -m "chore: add benchmark schema validation dependency"
```

Expected: Commit only when `git status --short pyproject.toml uv.lock` shows changes. If there are no changes, skip the commit.

## Task 6: Add GPU Result Formatter

**Files:**
- Create: `scripts/format_gpu_results.py`

- [ ] **Step 1: Write a failing formatter command**

Run:

```bash
python3 scripts/format_gpu_results.py --help
```

Expected: FAIL because the file does not exist.

- [ ] **Step 2: Add the formatter**

Create `scripts/format_gpu_results.py`:

```python
#!/usr/bin/env python3
"""Format GPU benchmark JSONL results as Markdown."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PREFERRED_BACKENDS = [
    "tenferro-cuda-eager",
    "tenferro-cuda-trace",
    "pytorch-cuda",
    "libtorch-cuda",
    "jax-cuda",
    "cublaslt",
    "cutlass",
    "cusolver",
    "cusparse",
    "ginkgo",
]

BACKEND_LABELS = {
    "tenferro-cuda-eager": "tenferro-rs CUDA eager",
    "tenferro-cuda-trace": "tenferro-rs CUDA trace",
    "pytorch-cuda": "PyTorch CUDA",
    "libtorch-cuda": "LibTorch CUDA",
    "jax-cuda": "JAX CUDA",
    "cublaslt": "cuBLASLt",
    "cutlass": "CUTLASS",
    "cusolver": "cuSOLVER",
    "cusparse": "cuSPARSE",
    "ginkgo": "Ginkgo",
}


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
    return records


def backend_order(records: list[dict[str, Any]]) -> list[str]:
    found = {record["backend"] for record in records}
    ordered = [backend for backend in PREFERRED_BACKENDS if backend in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def format_cell(record: dict[str, Any] | None) -> str:
    if record is None:
        return "-"
    status = record["status"]
    if status == "ok":
        median = record["timing"]["median_ms"]
        if median is None:
            return "ok"
        return f"{median:.3f}"
    if status == "not_configured":
        return "not configured"
    if status == "verification_failed":
        return "verification failed"
    if status == "cpu_fallback":
        return "CPU fallback"
    return status.replace("_", " ")


def format_markdown(records: list[dict[str, Any]]) -> str:
    if not records:
        return "# GPU Benchmark Results\n\nNo GPU benchmark records found.\n"

    backends = backend_order(records)
    by_suite_op: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_suite_op[(record["suite_id"], record["op"])].append(record)

    lines = ["# GPU Benchmark Results", ""]
    lines.append("Median time is reported in milliseconds for `ok` records.")
    lines.append("Non-`ok` cells show the structured backend status.")
    lines.append("")

    for (suite_id, op), group in sorted(by_suite_op.items()):
        lines.append(f"## {suite_id} / {op}")
        lines.append("")
        header = "| Problem | " + " | ".join(BACKEND_LABELS.get(b, b) for b in backends) + " |"
        separator = "|---|" + "|".join("---:" for _ in backends) + "|"
        lines.append(header)
        lines.append(separator)

        by_problem_backend: dict[tuple[str, str], dict[str, Any]] = {}
        problem_ids = sorted({record["problem_id"] for record in group})
        for record in group:
            by_problem_backend[(record["problem_id"], record["backend"])] = record

        for problem_id in problem_ids:
            row = [problem_id]
            for backend in backends:
                row.append(format_cell(by_problem_backend.get((problem_id, backend))))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    markdown = format_markdown(load_records(args.results))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Verify help works**

Run:

```bash
python3 scripts/format_gpu_results.py --help
```

Expected: PASS and prints usage.

- [ ] **Step 4: Commit**

Run:

```bash
git add scripts/format_gpu_results.py
git commit -m "feat: add GPU benchmark result formatter"
```

## Task 7: Add Phase 1 Smoke Orchestrator

**Files:**
- Create: `scripts/run_gpu_suite.sh`

- [ ] **Step 1: Write failing smoke command**

Run:

```bash
bash scripts/run_gpu_suite.sh
```

Expected: FAIL because the file does not exist.

- [ ] **Step 2: Add the orchestrator**

Create `scripts/run_gpu_suite.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/data/results"
REPORTS_DIR="$PROJECT_DIR/result"
TIMESTAMP="${GPU_BENCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

SUITES_VALUE="${GPU_BENCH_SUITE:-benchmarks/gpu/dense.yaml,benchmarks/gpu/einsum.yaml,benchmarks/gpu/sparse.yaml}"
BACKENDS_VALUE="${GPU_BENCH_BACKENDS:-tenferro-cuda-trace,tenferro-cuda-eager,pytorch-cuda,libtorch-cuda,jax-cuda,cublaslt,cutlass,cusolver,cusparse,ginkgo}"
DEVICE_ORDINAL="${GPU_BENCH_DEVICE:-0}"
PROBLEM_FILTER="${GPU_BENCH_PROBLEM:-}"

mkdir -p "$RESULTS_DIR" "$REPORTS_DIR"

IFS=',' read -r -a SUITES <<< "$SUITES_VALUE"
IFS=',' read -r -a BACKENDS <<< "$BACKENDS_VALUE"

echo "============================================"
echo " GPU benchmark contract suite"
echo "============================================"
echo "Suites:   $SUITES_VALUE"
echo "Backends: $BACKENDS_VALUE"
echo "Device:   cuda:$DEVICE_ORDINAL"
echo "Timestamp: $TIMESTAMP"
echo ""

VALIDATOR=(python3 "$SCRIPT_DIR/validate_benchmark_suite.py")
FORMATTER=(python3 "$SCRIPT_DIR/format_gpu_results.py")
if command -v uv >/dev/null 2>&1; then
    VALIDATOR=(uv run python "$SCRIPT_DIR/validate_benchmark_suite.py")
    FORMATTER=(uv run python "$SCRIPT_DIR/format_gpu_results.py")
fi

"${VALIDATOR[@]}" "${SUITES[@]}"

RESULT_JSONL="$RESULTS_DIR/gpu_contract_${TIMESTAMP}.jsonl"
python3 - "$RESULT_JSONL" "$DEVICE_ORDINAL" "$PROBLEM_FILTER" "${BACKENDS[@]}" -- "${SUITES[@]}" <<'PY'
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

output = Path(sys.argv[1])
device_ordinal = int(sys.argv[2])
problem_filter = sys.argv[3]
separator = sys.argv.index("--")
backends = sys.argv[4:separator]
suites = [Path(p) for p in sys.argv[separator + 1:]]


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


benchmark_commit = git_commit(Path.cwd())
tenferro_commit = git_commit(Path.cwd() / "extern" / "tenferro-rs")
timestamp = datetime.now(timezone.utc).isoformat()

with output.open("w") as fh:
    for suite_path in suites:
        suite = yaml.safe_load(suite_path.read_text())
        for problem in suite["problems"]:
            if problem_filter and problem["id"] != problem_filter:
                continue
            for backend in backends:
                if backend not in problem["backend_candidates"]:
                    status = "unsupported"
                    reason = f"{backend} is not listed for {problem['id']}"
                else:
                    status = "not_configured"
                    reason = "Phase 1 contract smoke does not execute GPU kernels"
                record = {
                    "schema_version": 1,
                    "suite_id": suite["suite_id"],
                    "problem_id": problem["id"],
                    "op": problem["op"],
                    "backend": backend,
                    "status": status,
                    "timing": {
                        "warmup_runs": 0,
                        "timed_runs": 0,
                        "compile_time_ms": None,
                        "first_run_ms": None,
                        "median_ms": None,
                        "min_ms": None,
                        "p95_ms": None,
                        "iqr_ms": None,
                        "timing_scope": "steady_state_host_api_plus_device_sync",
                    },
                    "performance": {
                        "tflops": None,
                        "effective_bandwidth_gbps": None,
                        "peak_memory_bytes": None,
                    },
                    "verification": {
                        "status": "skipped",
                        "reference_backend": None,
                        "max_abs_error": None,
                        "max_rel_error": None,
                        "residual": None,
                        "rtol": problem["verify"].get("rtol"),
                        "atol": problem["verify"].get("atol"),
                        "reason": reason,
                    },
                    "environment": {
                        "hostname": socket.gethostname(),
                        "timestamp_utc": timestamp,
                        "os": platform.platform(),
                        "gpu_name": None,
                        "gpu_uuid": None,
                        "gpu_memory_bytes": None,
                        "driver_version": None,
                        "cuda_version": None,
                        "cudnn_version": None,
                        "framework_version": None,
                        "tenferro_rs_commit": tenferro_commit,
                        "benchmark_repo_commit": benchmark_commit,
                        "env": {
                            "CUDA_PATH": os.environ.get("CUDA_PATH"),
                            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
                        },
                    },
                    "execution": {
                        "device": "cuda",
                        "device_ordinal": device_ordinal,
                        "execution_path": "phase1-contract-smoke",
                        "synchronization": "not executed",
                        "layout": json.dumps(problem.get("layout", {}), sort_keys=True),
                        "dtype": json.dumps(problem.get("dtype", {}), sort_keys=True),
                        "notes": "Generated by scripts/run_gpu_suite.sh Phase 1 smoke path.",
                        "unsupported_reason": reason,
                    },
                }
                fh.write(json.dumps(record, sort_keys=True) + "\n")
PY

"${VALIDATOR[@]}" --kind result "$RESULT_JSONL"

MARKDOWN_OUT="$RESULTS_DIR/gpu_results_${TIMESTAMP}.md"
REPORT_OUT="$REPORTS_DIR/gpu-benchmark-results.md"
"${FORMATTER[@]}" "$RESULT_JSONL" --output "$MARKDOWN_OUT"
cp "$MARKDOWN_OUT" "$REPORT_OUT"

echo ""
echo "GPU contract smoke complete"
echo "JSONL:    $RESULT_JSONL"
echo "Markdown: $MARKDOWN_OUT"
echo "Report:   $REPORT_OUT"
```

- [ ] **Step 3: Make the script executable**

Run:

```bash
chmod +x scripts/run_gpu_suite.sh
```

- [ ] **Step 4: Run smoke orchestrator**

Run:

```bash
GPU_BENCH_BACKENDS=tenferro-cuda-trace,pytorch-cuda bash scripts/run_gpu_suite.sh
```

Expected: PASS and prints paths for JSONL, Markdown, and `result/gpu-benchmark-results.md`.

- [ ] **Step 5: Validate generated outputs**

Run:

```bash
latest_jsonl="$(ls -t data/results/gpu_contract_*.jsonl | head -n 1)"
uv run python scripts/validate_benchmark_suite.py --kind result "$latest_jsonl"
rg -n "GPU Benchmark Results|tenferro-rs CUDA trace|PyTorch CUDA|not configured" result/gpu-benchmark-results.md
```

Expected: validation passes, and `rg` finds all listed terms.

- [ ] **Step 6: Commit**

Run:

```bash
git add scripts/run_gpu_suite.sh
git commit -m "feat: add GPU benchmark contract smoke runner"
```

## Task 8: Add Contract Test

**Files:**
- Create: `tests/test_gpu_benchmark_contract.sh`

- [ ] **Step 1: Write failing test command**

Run:

```bash
bash tests/test_gpu_benchmark_contract.sh
```

Expected: FAIL because the file does not exist.

- [ ] **Step 2: Add the test**

Create `tests/test_gpu_benchmark_contract.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

uv run python scripts/validate_benchmark_suite.py \
  benchmarks/gpu/dense.yaml \
  benchmarks/gpu/einsum.yaml \
  benchmarks/gpu/sparse.yaml

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

GPU_BENCH_TIMESTAMP=19990101_000000 \
GPU_BENCH_BACKENDS=tenferro-cuda-trace,pytorch-cuda \
GPU_BENCH_PROBLEM=dense_matmul_f64_256 \
  bash scripts/run_gpu_suite.sh

JSONL="data/results/gpu_contract_19990101_000000.jsonl"
MARKDOWN="data/results/gpu_results_19990101_000000.md"
REPORT="result/gpu-benchmark-results.md"

test -s "$JSONL"
test -s "$MARKDOWN"
test -s "$REPORT"

uv run python scripts/validate_benchmark_suite.py --kind result "$JSONL"

rg -n "GPU Benchmark Results" "$REPORT"
rg -n "dense_matmul_f64_256" "$REPORT"
rg -n "tenferro-rs CUDA trace" "$REPORT"
rg -n "PyTorch CUDA" "$REPORT"
rg -n "not configured" "$REPORT"
```

- [ ] **Step 3: Run the test**

Run:

```bash
bash tests/test_gpu_benchmark_contract.sh
```

Expected: PASS.

- [ ] **Step 4: Commit**

Run:

```bash
git add tests/test_gpu_benchmark_contract.sh
git commit -m "test: cover GPU benchmark contract smoke path"
```

## Task 9: Document GPU Contract Workflow

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Verify README does not mention the new workflow**

Run:

```bash
rg -n "GPU Benchmark Contract Workflow|run_gpu_suite" README.md
```

Expected: FAIL because the section is not present.

- [ ] **Step 2: Add README section after the Dev Container Workflow section**

Insert this Markdown after the existing Dev Container Workflow section and before `## Torch C++ Benchmark Workflow`:

```markdown
## GPU Benchmark Contract Workflow

The GPU benchmark layer uses shared suite and result schemas so tenferro-rs,
PyTorch Python, LibTorch C++, JAX, and vendor CUDA runners can report comparable
records. Phase 1 validates the contract and report generation without requiring
a GPU:

```bash
bash scripts/run_gpu_suite.sh
```

The smoke path emits structured `not_configured` records instead of measuring
kernels. This keeps schema and formatter tests runnable on CPU-only machines.
Measured CUDA runners are outside this Phase 1 plan and require separate follow-up implementation plans.

Validate suites and generated JSONL records manually:

```bash
uv run python scripts/validate_benchmark_suite.py \
  benchmarks/gpu/dense.yaml \
  benchmarks/gpu/einsum.yaml \
  benchmarks/gpu/sparse.yaml

uv run python scripts/validate_benchmark_suite.py --kind result \
  data/results/gpu_contract_*.jsonl
```

The latest generated GPU report is written to:

- `result/gpu-benchmark-results.md`
```

- [ ] **Step 3: Verify README section**

Run:

```bash
rg -n "GPU Benchmark Contract Workflow|run_gpu_suite|gpu-benchmark-results" README.md
```

Expected: PASS and prints matching lines.

- [ ] **Step 4: Commit**

Run:

```bash
git add README.md
git commit -m "docs: document GPU benchmark contract workflow"
```

## Task 10: Final Verification

**Files:**
- Read: `git status --short`

- [ ] **Step 1: Run existing non-GPU checks**

Run:

```bash
cargo metadata --no-deps --format-version 1 >/dev/null
bash tests/test_run_all_docs_outputs.sh
bash tests/test_extern_dependency_paths.sh
bash tests/test_run_all_rust_bin_selection.sh
bash tests/test_native_batch_layout_labels.sh
```

Expected: all commands PASS.

- [ ] **Step 2: Run new GPU contract test**

Run:

```bash
bash tests/test_gpu_benchmark_contract.sh
```

Expected: PASS.

- [ ] **Step 3: Inspect generated GPU report**

Run:

```bash
sed -n '1,160p' result/gpu-benchmark-results.md
```

Expected: report starts with `# GPU Benchmark Results`, includes `dense_matmul_f64_256`, and shows `not configured` for Phase 1 backend cells.

- [ ] **Step 4: Check working tree**

Run:

```bash
git status --short
```

Expected: only generated result files may be modified or untracked. Do not commit generated `data/results/gpu_contract_*.jsonl`, `data/results/gpu_results_*.md`, or `result/gpu-benchmark-results.md` in Phase 1 unless the project owner explicitly asks for generated benchmark artifacts.

- [ ] **Step 5: Commit any missed source changes**

Run:

```bash
git status --short
```

If source files from this plan are still unstaged, add and commit them with a message matching their task. If only generated result artifacts remain, leave them uncommitted.

## Self-Review Notes

Spec coverage:

- Suite schema, result schema, sample dense/einsum/sparse suites, validation, JSONL formatting, orchestration, and non-GPU testing are covered.
- Measured tenferro/PyTorch/LibTorch/JAX/vendor CUDA runners are intentionally deferred to Phase 2+ plans.
- Existing CPU scripts are not modified except README documentation.

Every file creation step includes full file content or exact insertion text. Every validation step has an expected result.
