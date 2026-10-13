# CUDA public API capability recheck

Raw probe: `data/verification/nvidia-gpu/gpu/linalg_capability/20261013_203719/`.

tenferro-rs: `fbd735abbfd5bcf0453f151472b274a71249f115`; harness binary source: `5f23b60`.

GPU1 diagnostic only. Previous unsupported declarations were removed before calling the current public direct APIs. No timings were collected. Runtime failures remain explicit failures and do not count as numerical or measured operation coverage.

| Case | Actual status | Observation |
|---|---|---|
| inv_f64_2x2 | runtime_failed | broadcast_in_dim: runtime state failure: expected CubeCL GPU tensor, got host tensor. Use upload_tensor() to transfer to GPU before calling GPU ops. |
| pinv_with_rtol_f64_2x2 | runtime_failed | mul: runtime state failure: expected CubeCL GPU tensor view, got host tensor. Use upload_tensor() to transfer to GPU before calling GPU ops. |

Current source explains these direct-path failures: inv_owned calls eye_like, which passes scalar_real host storage to broadcast; pinv_from_svd passes a scalar_real host threshold to binary_mul. Neither public call accepts a replacement device scalar. Eager linalg routes have separate implementations and are not classified by this direct-only probe.
