# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-11-21T08:23:11+00:00`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-98d8618e-77c8-160a-92f2-117e34fd43df`
- Memory: `80 GiB`
- Driver version: `595.112.07`
- CUDA version: `13.2`
- CUDA runtime: `12.8`
- cuDNN version: `91900`
- Container: `devcontainer`
- CUDA toolkit: `unknown`
- CUDA toolkit root: `/opt/tenferro-benchmark-ci/cuda`
- cuTENSOR version: `2.2.0`
- cuSOLVER version: `11.7.3`

`tenferro-cuda-transpose` is the eager `TensorStructural::transpose` op on `CudaBackend` (compact col-major input only); `tenferro-cuda-to-contiguous` is `TypedTensor::backend_region_view` (source layout) + `TypedTensorView::transpose_view(perm)` + `TensorViewCanonicalization::to_contiguous` (accepts arbitrary source strides). For framework comparisons, `tenferro-cuda-to-contiguous` is the primary like-for-like column for PyTorch's view/permute-then-materialize semantics; allocation and reuse timings are shown separately. `tenferro-cuda-transpose` is the direct structural-permutation comparison for primitive/kernel-oriented backends such as cuTENSOR. Both public tenferro columns allocate a fresh device tensor on every call. `tenferro-cuda-destination-reuse` calls the public `CudaBackend::copy_read_into` override from a compact source into an inverse-permuted caller-owned destination view allocated once outside timing. The production path uses the native CUDA copy kernel, so it is distinct from the raw cuTENSOR control. Its per-call destination-view metadata construction is included in the timed public API dispatch. `cutensor`, `pytorch-cuda`, and `memcpy-d2d` also reuse a destination buffer allocated once per pattern. `memcpy-d2d` only participates in the contiguous identity-permutation baseline pattern. Correctness is verified against a host-computed naive reference, downloaded outside any timed region, before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern. A `skipped` cell means the backend does not participate in that pattern's semantics (or, for `cutensor`, that the installed cuTENSOR library rejected the pattern's rank at runtime) -- both are reported as `skipped` rather than a failure.

The GPU-only pattern set is intentionally larger than the CPU permutation set: most rows contain 2^29 f64 elements (4 GiB per tensor) so A100-class measurements exercise steady-state device throughput at roughly the 10 ms scale instead of launch/synchronization overhead.

Device-copy baseline: `memcpy-d2d` median 5.142 ms, 1670.65 GB/s for `memcpy_24d_64x2`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

## Allocating output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) |
|---|---|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | **9.867 (9.865 / 9.869)** | 9.870 (9.868 / 9.874) |
| `reverse_18d_3` | 18D 3^18 reverse | 30.190 (30.184 / 30.197) | **30.170 (30.168 / 30.176)** |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 42.018 (41.992 / 42.053) | **41.988 (41.971 / 42.704)** |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 7.254 (7.068 / 7.270) | **7.213 (7.123 / 7.264)** |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | **6.095 (6.055 / 6.119)** | 6.157 (6.137 / 6.199) |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | **8.325 (8.322 / 8.330)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | **5.585 (5.584 / 5.589)** | 5.729 (5.725 / 5.732) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | **5.280 (5.278 / 5.288)** | 5.405 (5.402 / 5.409) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | **5.209 (5.207 / 5.213)** | 5.314 (5.312 / 5.317) |

## Reusing output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA copy_read_into (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 9.866 (9.863 / 9.869) | **6.314 (6.051 / 6.317)** | 7.361 (7.359 / 7.364) |
| `reverse_18d_3` | 18D 3^18 reverse | 61.209 (61.046 / 61.404) | **8.617 (8.556 / 8.618)** | 33.207 (33.206 / 33.215) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 85.077 (84.972 / 85.246) | **7.595 (7.593 / 7.596)** | 45.691 (45.686 / 45.728) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 7.200 (7.091 / 7.263) | **5.343 (5.339 / 5.347)** | 5.944 (5.938 / 5.949) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 6.093 (6.052 / 6.142) | 5.533 (5.433 / 5.579) | **5.260 (5.258 / 5.262)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | 5.534 (5.417 / 5.578) | **5.402 (5.398 / 5.404)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 5.574 (5.572 / 5.578) | **5.520 (5.516 / 5.522)** | 20.852 (20.849 / 20.856) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 5.263 (5.260 / 5.267) | **5.234 (5.229 / 5.239)** | 7.461 (7.409 / 7.473) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | **5.181 (5.178 / 5.189)** | 5.320 (5.317 / 5.323) | 27.523 (27.520 / 27.533) |
