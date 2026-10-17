# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-10-17T10:55:14.127799+00:00`
- tenferro-rs commit: `88daa885a5bb09e07f3031814bbb1a21f7a74e62`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-530977e1-4968-9283-4129-9fbec3e66542`
- Memory: `80 GiB`
- Driver version: `580.126.09`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92700`

`tenferro-cuda-transpose` is the eager `TensorStructural::transpose` op on `CudaBackend` (compact col-major input only); `tenferro-cuda-to-contiguous` is `TypedTensor::backend_region_view` (source layout) + `TypedTensorView::transpose_view(perm)` + `TensorViewCanonicalization::to_contiguous` (accepts arbitrary source strides). For framework comparisons, `tenferro-cuda-to-contiguous` is the primary like-for-like column for PyTorch's view/permute-then-materialize semantics; allocation and reuse timings are shown separately. `tenferro-cuda-transpose` is the direct structural-permutation comparison for primitive/kernel-oriented backends such as cuTENSOR. Both public tenferro columns allocate a fresh device tensor on every call. `tenferro-cuda-destination-reuse` calls the public `CudaBackend::copy_read_into` override from a compact source into an inverse-permuted caller-owned destination view allocated once outside timing. The production path uses the native CUDA copy kernel, so it is distinct from the raw cuTENSOR control. Its per-call destination-view metadata construction is included in the timed public API dispatch. `cutensor`, `pytorch-cuda`, and `memcpy-d2d` also reuse a destination buffer allocated once per pattern. `memcpy-d2d` only participates in the contiguous identity-permutation baseline pattern. Correctness is verified against a host-computed naive reference, downloaded outside any timed region, before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern. A `skipped` cell means the backend does not participate in that pattern's semantics (or, for `cutensor`, that the installed cuTENSOR library rejected the pattern's rank at runtime) -- both are reported as `skipped` rather than a failure.

The GPU-only pattern set is intentionally larger than the CPU permutation set: most rows contain 2^29 f64 elements (4 GiB per tensor) so A100-class measurements exercise steady-state device throughput at roughly the 10 ms scale instead of launch/synchronization overhead.

Device-copy baseline: `memcpy-d2d` median 5.133 ms, 1673.46 GB/s for `memcpy_24d_64x2`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

## Allocating output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) |
|---|---|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 6.324 (6.123 / 6.326) | **6.316 (6.181 / 6.317)** |
| `reverse_18d_3` | 18D 3^18 reverse | 8.720 (8.623 / 8.725) | **8.712 (8.617 / 8.718)** |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | **7.628 (7.627 / 7.631)** | 8.271 (8.269 / 8.272) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | **5.404 (5.402 / 5.408)** | 5.557 (5.553 / 5.563) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | **5.567 (5.494 / 5.606)** | 5.646 (5.608 / 5.679) |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | **5.603 (5.437 / 5.658)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | **5.571 (5.569 / 5.573)** | 7.007 (7.000 / 7.010) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | **5.308 (5.306 / 5.316)** | 5.413 (5.409 / 5.416) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | **5.397 (5.393 / 5.402)** | 6.903 (6.900 / 6.904) |

## Reusing output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA copy_read_into (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 6.311 (6.108 / 6.312) | **6.210 (6.011 / 6.280)** | 7.355 (7.354 / 7.360) |
| `reverse_18d_3` | 18D 3^18 reverse | **8.628 (8.621 / 8.781)** | 8.743 (8.618 / 8.751) | 33.189 (33.186 / 33.195) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 7.615 (7.614 / 7.619) | **7.584 (7.583 / 7.584)** | 45.773 (45.770 / 45.775) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 5.357 (5.354 / 5.360) | **5.330 (5.329 / 5.336)** | 5.936 (5.930 / 5.938) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 5.554 (5.450 / 5.595) | 5.523 (5.420 / 5.563) | **5.260 (5.258 / 5.262)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | 5.523 (5.423 / 5.566) | **5.402 (5.398 / 5.404)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 5.527 (5.523 / 5.536) | **5.508 (5.503 / 5.512)** | 20.778 (20.772 / 20.781) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 5.261 (5.254 / 5.265) | **5.233 (5.232 / 5.239)** | 7.497 (7.447 / 7.499) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | 5.338 (5.333 / 5.342) | **5.313 (5.310 / 5.319)** | 27.394 (27.392 / 27.396) |

Correctness gate: exact **sampled** verification. `sampled` checks a fixed set of per-axis corners plus roughly 1M seeded output elements against the host source index mapping; `full` downloads the whole output and compares it elementwise against a full host transpose. Set `PERMUTATION_FULL_VERIFY=1` to request the full gate.
