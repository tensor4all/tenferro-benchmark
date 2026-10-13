# CUDA public API capability recheck

Raw probe: `data/verification/nvidia-gpu/gpu/linalg_capability/20261013_200509/`.

tenferro-rs: `dfbdf6b66d40162f4731da116ec42ed034da0dce`; harness binary source: `8ee736ea199dc5d7b9ee5702526f931b3b934807`.

GPU1 diagnostic only. Previous unsupported declarations were removed before calling the current public direct APIs. No timings were collected. Runtime failures remain explicit failures and do not count as numerical or measured operation coverage.

| Case | Actual status | Observation |
|---|---|---|
| pinv_f32_32x32 | runtime_failed | mul: runtime state failure: expected CubeCL GPU tensor view, got host tensor. Use upload_tensor() to transfer to GPU before calling GPU ops. |
| pinv_f64_32x32 | runtime_failed | mul: runtime state failure: expected CubeCL GPU tensor view, got host tensor. Use upload_tensor() to transfer to GPU before calling GPU ops. |
| inv_f32_32x32 | runtime_failed | broadcast_in_dim: runtime state failure: expected CubeCL GPU tensor, got host tensor. Use upload_tensor() to transfer to GPU before calling GPU ops. |
| inv_f64_32x32 | runtime_failed | broadcast_in_dim: runtime state failure: expected CubeCL GPU tensor, got host tensor. Use upload_tensor() to transfer to GPU before calling GPU ops. |
| pinv_f32_512x512 | runtime_failed | mul: runtime state failure: expected CubeCL GPU tensor view, got host tensor. Use upload_tensor() to transfer to GPU before calling GPU ops. |
| pinv_f64_512x512 | runtime_failed | mul: runtime state failure: expected CubeCL GPU tensor view, got host tensor. Use upload_tensor() to transfer to GPU before calling GPU ops. |
| inv_f32_512x512 | runtime_failed | broadcast_in_dim: runtime state failure: expected CubeCL GPU tensor, got host tensor. Use upload_tensor() to transfer to GPU before calling GPU ops. |
| inv_f64_512x512 | runtime_failed | broadcast_in_dim: runtime state failure: expected CubeCL GPU tensor, got host tensor. Use upload_tensor() to transfer to GPU before calling GPU ops. |

Current source explains these direct-path failures: inv_owned calls eye_like, which passes scalar_real host storage to broadcast; pinv_from_svd passes a scalar_real host threshold to binary_mul. Neither public call accepts a replacement device scalar. Eager linalg routes have separate implementations and are not classified by this direct-only probe.
