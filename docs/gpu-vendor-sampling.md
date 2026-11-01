# GPU vendor reference sampling

The full core matrix keeps CUTLASS, Ginkgo and the PyTorch vendor-preference
arms. These paths use the same operation-only timing boundary and bounded
sampler as the framework paths. Notes retain raw batch milliseconds,
operations/sample, setup/precision and available case phases. Provider notes
are merged into this structured metadata; they never replace the samples.

CUTLASS uses persistent F64 `Gemm` or `GemmBatched` operators. Input layouts,
fixed arguments, capability checks, initialization and workspace preparation
are outside timing. A timed call allocates its fresh output, updates only the
output pointers and launches the native operator. Batched GEMM is one native
batched call, rather than a loop of GEMMs followed by a stack. Output storage
is uninitialized because beta=0 does not require its previous values.

Ginkgo keeps its executor, CSR operator and input Dense descriptor outside
timing. The returning-output wrapper allocates a fresh Tensor and its native
Dense output descriptor as part of the declared allocating operation. Both
objects remain alive through native executor synchronization and clock stop;
only then are they destroyed. Inputs use native Ginkgo I32 CSR indices,
prepared outside timing. This is not a raw output-reuse API comparison.
PyTorch's sparse reference uses its native CSR indices and public `mv`/`mm`;
there is no timed squeeze/stack composition.

The `cublaslt` backend ID selects PyTorch's explicit cuBLASLt preference.
PyTorch may still fall back when the preferred library cannot implement an
operation, so reports say **PyTorch (cuBLASLt preferred)** rather than asserting
a raw cuBLASLt call. The cuSOLVER and cuSPARSE columns also name their actual
PyTorch wrappers. Their metadata records the configuration and preserves the
underlying sampler fields. These legacy IDs remain selectable in full.

The configured output-retention budget is 256 MiB. Some existing workloads
return an intrinsic 512 MiB output. Those cases use one operation per sample
and explicitly record a larger effective bound for that single output; inputs
are not shrunk. The report distinguishes configured and effective bounds.
These are output-retention bounds, not total device-memory limits.

Untimed checks on physical GPU1 validate dtype/device, numerical values and
multiple simultaneously retained native outputs:

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'CUDA_VISIBLE_DEVICES=GPU-ed896702-ca5c-93f6-2537-06fffdac4a91 \
    GPU_BENCH_PHYSICAL_DEVICE=1 GPU_BENCH_VERIFY_ONLY=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 MATMUL_NUM_THREADS=1 \
    MAX_JOBS=2 .venv/bin/python scripts/verify_gpu_vendors.py \
    --output data/verification/nvidia-gpu/gpu/vendor_sampling/<timestamp>'
```

Use the local card UUID on other hosts. The ten small probe rows are numerical
checks, not coverage or timings of the full-size workloads. Then collect the
routine set sequentially on physical GPU0 with the vendor arms enabled
(`BENCH_ROUTINE_VENDOR_ARMS=1`) and the desired independent effort setting.
Native extension build failures are runtime failures with the actual compiler
error, not misreported missing SDKs.
