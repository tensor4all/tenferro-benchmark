# gpu/reuse_contraction: CUDA public API benchmark

Measured cells: **150**. Unsupported cells: **74**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1413.2 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261105_0820/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Transposed borrowed-read cases use prepared native CUDA views in tenferro and PyTorch. JAX compares the same logical values in its native representation; its columns do not claim an equivalent public strided-view contract. View creation and input conversion are outside timing.

Generalized batch dot preserves native output order: tenferro [m,n,b], PyTorch/JAX [b,m,n]. Input permutation views and dimension-number configurations are prepared before timing. Output axes are normalized only on downloaded CPU arrays for verification; no timed output transpose or copy is added.

Prepared einsum uses the public ConcreteEinsumPlan for ik,kj->ij, built once outside execution and sampling. Native matrix multiplication/reuse/accumulation references match the selected mathematical and output-ownership contracts; they do not assert an equivalent prepared-plan interface. Transposed input views/values, read/write handles and accumulation seed restoration are untimed.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| add_into_f64_n1048576 | 21.678 (0.250) | unsupported | 12.605 (0.126) | unsupported |
| add_read_into_f64_n1048576 | 21.766 (0.213) | unsupported | 12.599 (0.120) | unsupported |
| sub_into_f64_n1048576 | 21.584 (0.183) | unsupported | 12.645 (0.140) | unsupported |
| sub_read_into_f64_n1048576 | 21.685 (0.138) | unsupported | 12.673 (0.062) | unsupported |
| mul_into_f64_n1048576 | 21.553 (0.157) | unsupported | 12.832 (0.102) | unsupported |
| mul_read_into_f64_n1048576 | 21.492 (0.152) | unsupported | 12.795 (0.089) | unsupported |
| div_into_f64_n1048576 | 21.425 (0.177) | unsupported | 13.368 (0.120) | unsupported |
| div_read_into_f64_n1048576 | 21.372 (0.214) | unsupported | 13.309 (0.179) | unsupported |
| neg_into_f64_n1048576 | 17.552 (0.150) | unsupported | 7.835 (0.055) | unsupported |
| neg_read_into_f64_n1048576 | 17.616 (0.121) | unsupported | 7.860 (0.031) | unsupported |
| conj_into_c64_n1048576 | 33.480 (0.246) | unsupported | 25.321 (0.333) | unsupported |
| conj_read_into_c64_n1048576 | 33.547 (0.129) | unsupported | 25.314 (0.279) | unsupported |
| tensordot_f64_n262144 | 41.788 (0.365) | 45.511 (0.219) | 25.191 (0.305) | 194.178 (20.497) |
| vdot_read_f64_n262144 | 87.984 (0.716) | unsupported | 9.807 (0.015) | 70.118 (10.157) |
| norm_squared_read_f64_n262144 | 48.861 (0.578) | unsupported | 9.534 (0.012) | 61.397 (3.085) |
| axpby_read_into_accum_f64_n1048576 | 31.360 (0.140) | unsupported | 16.306 (0.221) | unsupported |
| dot_general_read_into_f64_s256 | 17.414 (0.257) | unsupported | 9.785 (0.054) | unsupported |
| dot_general_read_into_accum_f64_s256 | 19.396 (0.332) | unsupported | 10.953 (0.178) | unsupported |
| dot_general_read_into_accum_cached_f64_s256 | 20.299 (1.247) | unsupported | 11.046 (0.068) | unsupported |
| add_strided_read_f64_n1048576 | 106.584 (1.055) | unsupported | 12.632 (0.086) | 64.689 (1.746) |
| mul_strided_read_f64_n1048576 | 107.264 (3.724) | unsupported | 12.784 (0.060) | 59.705 (0.624) |
| neg_strided_read_f64_n1048576 | 63.889 (3.868) | unsupported | 7.916 (0.082) | 58.389 (0.907) |
| exp_strided_read_f64_n1048576 | 63.132 (0.200) | unsupported | 8.357 (0.041) | 46.976 (0.529) |
| tanh_strided_read_f64_n1048576 | 64.973 (3.527) | unsupported | 8.195 (0.057) | 51.174 (0.715) |
| reduce_sum_strided_read_f64_n1048576 | 100.705 (1.069) | unsupported | 13.307 (0.015) | 53.114 (14.753) |
| conj_strided_read_c64_n1048576 | 91.418 (0.610) | unsupported | 25.407 (0.182) | 51.592 (0.895) |
| dot_general_batch_f64_s256 | 26.368 (0.224) | 28.874 (0.084) | 20.913 (0.203) | 89.836 (2.165) |
| dot_conj_l0r0_c64_n262144 | 36.130 (0.820) | 36.497 (0.161) | 12.941 (0.012) | 56.201 (0.996) |
| einsum_plan_owned_f64_s256 | 23.499 (0.338) | unsupported | 10.231 (0.132) | 54.437 (0.707) |
| einsum_plan_read_f64_s256 | 23.300 (0.172) | unsupported | 10.170 (0.065) | 54.168 (0.998) |
| einsum_plan_strided_read_f64_s256 | 68.132 (0.539) | unsupported | 10.320 (0.081) | 54.679 (0.486) |
| einsum_plan_into_f64_s256 | 16.170 (0.208) | unsupported | 8.803 (0.174) | unsupported |
| einsum_plan_read_into_f64_s256 | 15.945 (0.398) | unsupported | 8.695 (0.222) | unsupported |
| einsum_plan_read_into_accum_f64_s256 | 17.910 (0.103) | unsupported | 10.916 (0.050) | unsupported |
| einsum_plan_strided_read_into_f64_s256 | 60.541 (0.844) | unsupported | 8.886 (0.089) | unsupported |
| einsum_plan_strided_read_into_accum_f64_s256 | 62.064 (0.443) | unsupported | 11.097 (0.046) | unsupported |
| sub_strided_read_f64_n1048576 | 105.262 (0.384) | unsupported | 12.787 (0.155) | 62.896 (1.056) |
| div_strided_read_f64_n1048576 | 104.518 (0.311) | unsupported | 13.277 (0.211) | 53.703 (0.903) |
| rem_strided_read_f64_n1048576 | 128.793 (2.367) | unsupported | 12.306 (0.156) | 52.796 (0.726) |
| pow_strided_read_f64_n1048576 | 131.877 (0.915) | unsupported | 29.079 (0.032) | 63.155 (0.834) |
| maximum_strided_read_f64_n1048576 | 118.115 (1.062) | unsupported | 12.162 (0.201) | 60.108 (0.896) |
| minimum_strided_read_f64_n1048576 | 118.448 (0.635) | unsupported | 12.127 (0.146) | 59.847 (0.682) |
| abs_strided_read_f64_n1048576 | 61.910 (0.196) | unsupported | 7.888 (0.070) | 50.812 (0.610) |
| sign_strided_read_f64_n1048576 | 62.204 (0.158) | unsupported | 7.802 (0.057) | 52.594 (0.545) |
| expm1_strided_read_f64_n1048576 | 64.114 (0.386) | unsupported | 9.908 (0.050) | 53.119 (0.442) |
| log_strided_read_f64_n1048576 | 66.503 (0.222) | unsupported | 12.612 (0.043) | 53.482 (0.954) |
| log1p_strided_read_f64_n1048576 | 66.532 (0.293) | unsupported | 13.061 (0.042) | 50.385 (0.708) |
| erf_strided_read_f64_n1048576 | 68.820 (0.310) | unsupported | 14.347 (0.062) | 69.973 (0.570) |
| sin_strided_read_f64_n1048576 | 65.406 (0.286) | unsupported | 10.075 (0.037) | 51.408 (0.295) |
| cos_strided_read_f64_n1048576 | 65.769 (0.531) | unsupported | 10.145 (0.063) | 47.981 (3.816) |
| sqrt_strided_read_f64_n1048576 | 62.631 (0.258) | unsupported | 7.876 (0.058) | 47.416 (0.502) |
| rsqrt_strided_read_f64_n1048576 | 62.693 (0.375) | unsupported | 7.882 (0.047) | 73.246 (1.067) |
| reduce_prod_strided_read_f64_n1048576 | 98.526 (0.734) | unsupported | 13.324 (0.015) | 52.102 (0.756) |
| reduce_max_strided_read_f64_n1048576 | 99.733 (0.941) | unsupported | 13.390 (0.010) | 51.645 (1.566) |
| reduce_min_strided_read_f64_n1048576 | 100.207 (0.447) | unsupported | 13.431 (0.010) | 47.055 (0.776) |
| reduce_sum_squares_strided_read_f64_n1048576 | 98.900 (0.445) | unsupported | 21.637 (0.026) | 845.528 (5.845) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.626 | 16777216 | 0.234 | 0 |
| abs_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.252 | 16777216 | 1.505 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.981 | 16777216 | 0.192 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| add_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.403 | 25165824 | 1.413 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.694 | 25165824 | 0.821 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.403 | 25165824 | 1.413 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.697 | 25165824 | 0.818 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.070 | 25165824 | 0.275 | 0 |
| add_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.404 | 25165824 | 1.410 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.411 | 25165824 | 0.167 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| axpby_read_into_accum_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.522 | 25165824 | 1.092 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.004 | 25165824 | 0.568 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No matching public caller-owned accumulating output API; donation is not output reuse. |
| conj_into_c64_n1048576 | jax-cuda | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.405 | 33554432 | 0.938 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.536 | 33554432 | 0.709 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1048576 | jax-cuda | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.405 | 33554432 | 0.938 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.537 | 33554432 | 0.708 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_strided_read_c64_n1048576 | jax-cuda | 16 | 15 | 0.825 | 33554432 | 0.460 | 0 |
| conj_strided_read_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.407 | 33554432 | 0.935 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.463 | 33554432 | 0.260 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| cos_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.535 | 16777216 | 0.247 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.325 | 16777216 | 1.170 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.105 | 16777216 | 0.181 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| div_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.428 | 25165824 | 1.332 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.686 | 25165824 | 0.831 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.426 | 25165824 | 1.338 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.684 | 25165824 | 0.833 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.719 | 25165824 | 0.332 | 0 |
| div_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.425 | 25165824 | 1.341 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.345 | 25165824 | 0.170 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| dot_conj_l0r0_c64_n262144 | jax-cuda | 64 | 15 | 3.597 | 8388624 | 0.106 | 1.221e-05 |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | 256 | 15 | 3.313 | 8388624 | 0.459 | 1.366e-05 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | 64 | 15 | 2.312 | 8388624 | 0.164 | 1.212e-05 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | 64 | 15 | 2.336 | 8388624 | 0.163 | 1.212e-05 |
| dot_general_batch_f64_s256 | jax-cuda | 32 | 15 | 2.875 | 3210240 | 0.025 | 7.463e-07 |
| dot_general_batch_f64_s256 | pytorch-cuda | 128 | 15 | 2.677 | 3210240 | 0.109 | 7.463e-07 |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | 128 | 15 | 3.375 | 3210240 | 0.086 | 7.463e-07 |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | 128 | 15 | 3.696 | 3210240 | 0.079 | 7.463e-07 |
| dot_general_read_into_accum_cached_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | 256 | 15 | 2.828 | 2097152 | 0.134 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.598 | 2097152 | 0.073 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.804 | 2097152 | 0.135 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.483 | 2097152 | 0.077 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.505 | 2097152 | 0.152 | 0 |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.229 | 2097152 | 0.085 | 0 |
| dot_general_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| einsum_plan_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.254 | 1070080 | 0.086 | 0 |
| einsum_plan_into_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.070 | 1070080 | 0.047 | 0 |
| einsum_plan_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_owned_f64_s256 | jax-cuda | 64 | 15 | 3.484 | 1070080 | 0.014 | 0 |
| einsum_plan_owned_f64_s256 | pytorch-cuda | 256 | 15 | 2.619 | 1070080 | 0.074 | 0 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-direct | 128 | 15 | 3.008 | 1070080 | 0.032 | 0 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_f64_s256 | jax-cuda | 64 | 15 | 3.467 | 1070080 | 0.014 | 0 |
| einsum_plan_read_f64_s256 | pytorch-cuda | 256 | 15 | 2.604 | 1070080 | 0.074 | 0 |
| einsum_plan_read_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.982 | 1070080 | 0.032 | 0 |
| einsum_plan_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_accum_f64_s256 | jax-cuda | — | — | — | 1608704 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.795 | 1608704 | 0.104 | 0 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.292 | 1608704 | 0.064 | 0 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.226 | 1070080 | 0.087 | 0 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.041 | 1070080 | 0.047 | 0 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_f64_s256 | jax-cuda | 64 | 15 | 3.499 | 1070080 | 0.014 | 0 |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | 256 | 15 | 2.642 | 1070080 | 0.073 | 0 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-direct | 32 | 15 | 2.180 | 1070080 | 0.011 | 0 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_accum_f64_s256 | jax-cuda | — | — | — | 1608704 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.841 | 1608704 | 0.103 | 0 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-direct | 32 | 15 | 1.986 | 1608704 | 0.018 | 0 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.275 | 1070080 | 0.085 | 0 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-direct | 32 | 15 | 1.937 | 1070080 | 0.013 | 0 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| erf_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.239 | 16777216 | 0.170 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.459 | 16777216 | 0.827 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.202 | 16777216 | 0.173 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| exp_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.503 | 16777216 | 0.253 | 1.095e-07 |
| exp_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.267 | 16777216 | 1.421 | 1.095e-07 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.020 | 16777216 | 0.188 | 1.095e-07 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| expm1_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.700 | 16777216 | 0.223 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.317 | 16777216 | 1.198 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.052 | 16777216 | 0.185 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log1p_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.612 | 16777216 | 0.236 | 1.11e-07 |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.418 | 16777216 | 0.909 | 1.11e-07 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.129 | 16777216 | 0.178 | 1.11e-07 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.711 | 16777216 | 0.222 | 4.308e-08 |
| log_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.404 | 16777216 | 0.941 | 4.308e-08 |
| log_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.128 | 16777216 | 0.179 | 4.308e-08 |
| log_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| maximum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.923 | 25165824 | 0.296 | 0 |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.389 | 25165824 | 1.464 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.780 | 25165824 | 0.151 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| minimum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.915 | 25165824 | 0.298 | 0 |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.388 | 25165824 | 1.468 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.790 | 25165824 | 0.150 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| mul_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.411 | 25165824 | 1.388 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.690 | 25165824 | 0.826 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.409 | 25165824 | 1.392 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.688 | 25165824 | 0.829 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.911 | 25165824 | 0.298 | 0 |
| mul_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.409 | 25165824 | 1.393 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.432 | 25165824 | 0.166 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| neg_into_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.251 | 16777216 | 1.515 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.562 | 16777216 | 0.676 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.252 | 16777216 | 1.510 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.564 | 16777216 | 0.674 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.868 | 16777216 | 0.203 | 0 |
| neg_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.253 | 16777216 | 1.500 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.044 | 16777216 | 0.186 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| norm_squared_read_f64_n262144 | jax-cuda | 32 | 15 | 1.965 | 2097160 | 0.024 | 1.664e-07 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.441 | 2097160 | 0.156 | 1.664e-07 |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | 64 | 15 | 3.127 | 2097160 | 0.030 | 1.664e-07 |
| norm_squared_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 2097160 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| pow_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.021 | 25165824 | 0.282 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.931 | 25165824 | 0.612 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.110 | 25165824 | 0.135 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 3.305 | 8392704 | 0.115 | 0 |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.428 | 8392704 | 0.444 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.191 | 8392704 | 0.060 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 3.012 | 8392704 | 0.126 | 0 |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.438 | 8392704 | 0.442 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.207 | 8392704 | 0.059 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 3.335 | 8392704 | 0.114 | 4.995e-06 |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.411 | 8392704 | 0.446 | 4.106e-06 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.153 | 8392704 | 0.060 | 4.661e-06 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | 4 | 15 | 3.382 | 8392704 | 0.007 | 3.311e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.770 | 8392704 | 0.274 | 3.309e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.165 | 8392704 | 0.060 | 3.312e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.700 | 8392704 | 0.112 | 7.159e-06 |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.407 | 8392704 | 0.446 | 5.935e-06 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.223 | 8392704 | 0.059 | 4.724e-06 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rem_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.689 | 25165824 | 0.337 | 0 |
| rem_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.394 | 25165824 | 1.447 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.061 | 25165824 | 0.138 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.344 | 16777216 | 0.162 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.252 | 16777216 | 1.506 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.006 | 16777216 | 0.189 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sign_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.683 | 16777216 | 0.226 | 0 |
| sign_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.250 | 16777216 | 1.522 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.991 | 16777216 | 0.191 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sin_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.645 | 16777216 | 0.231 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.322 | 16777216 | 1.178 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.093 | 16777216 | 0.182 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sqrt_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.517 | 16777216 | 0.250 | 0 |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.252 | 16777216 | 1.507 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.004 | 16777216 | 0.190 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sub_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.405 | 25165824 | 1.408 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.691 | 25165824 | 0.825 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.406 | 25165824 | 1.405 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.694 | 25165824 | 0.821 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.013 | 25165824 | 0.283 | 0 |
| sub_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.409 | 25165824 | 1.393 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.368 | 25165824 | 0.169 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tanh_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.638 | 16777216 | 0.232 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.262 | 16777216 | 1.449 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.079 | 16777216 | 0.183 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tensordot_f64_n262144 | jax-cuda | 16 | 15 | 3.107 | 6291456 | 0.023 | 6.714e-06 |
| tensordot_f64_n262144 | pytorch-cuda | 128 | 15 | 3.224 | 6291456 | 0.177 | 6.714e-06 |
| tensordot_f64_n262144 | tenferro-cuda-direct | 64 | 15 | 2.674 | 6291456 | 0.107 | 6.714e-06 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 64 | 15 | 2.913 | 6291456 | 0.098 | 6.714e-06 |
| vdot_read_f64_n262144 | jax-cuda | 32 | 15 | 2.244 | 4194312 | 0.042 | 2.995e-06 |
| vdot_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.511 | 4194312 | 0.303 | 2.995e-06 |
| vdot_read_f64_n262144 | tenferro-cuda-direct | 32 | 15 | 2.816 | 4194312 | 0.034 | 2.995e-06 |
| vdot_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 4194312 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: /home/shinaoka/work/wallclock/data/results/nvidia-gpu/gpu/routine_selection/20261105_0820/gpu_reuse_contraction.yaml
timestamp: '2026-11-05T08:02:00+00:00'
tenferro_rs:
  path: /home/shinaoka/work/wallclock/extern/tenferro-rs
  commit: a000e5c4929d5bebb91e783c329b27106baed12e
  dirty: false
  features:
  - cuda
  resolved_path: /home/shinaoka/work/wallclock/extern/tenferro-rs
harness:
  path: /home/shinaoka/work/wallclock
  commit: 18ae0c54c3f6cc4493988efe87a65148334a5c77
  dirty: false
resolved_deps:
  strided-perm:
  - 0.8.0
  - 0.10.0
  strided-view:
  - 0.7.0
  threadteam:
  - 0.1.0
  - 0.3.0
host:
  cpu: AMD EPYC 7713P 64-Core Processor
  logical_cpus: 64
  power:
    cpufreq_boost: '1'
    governor: schedutil
    scaling_max_freq_khz: '2000000'
    smt:
      active: '0'
      control: notsupported
  numa:
    nodes: 1
    cpulists:
      '0': 0-63
  smt_siblings:
  - '0'
  - '1'
  - '2'
  - '3'
  - '4'
  - '5'
  - '6'
  - '7'
  - '8'
  - '9'
  - '10'
  - '11'
  - '12'
  - '13'
  - '14'
  - '15'
  - '16'
  - '17'
  - '18'
  - '19'
  - '20'
  - '21'
  - '22'
  - '23'
  - '24'
  - '25'
  - '26'
  - '27'
  - '28'
  - '29'
  - '30'
  - '31'
  - '32'
  - '33'
  - '34'
  - '35'
  - '36'
  - '37'
  - '38'
  - '39'
  - '40'
  - '41'
  - '42'
  - '43'
  - '44'
  - '45'
  - '46'
  - '47'
  - '48'
  - '49'
  - '50'
  - '51'
  - '52'
  - '53'
  - '54'
  - '55'
  - '56'
  - '57'
  - '58'
  - '59'
  - '60'
  - '61'
  - '62'
  - '63'
  l3_domains:
  - 0-7
  - 8-15
  - 16-23
  - 24-31
  - 32-39
  - 40-47
  - 48-55
  - 56-63
build:
  toolchain:
    rustc: 'rustc 1.99.1 (58cf8f9c3 2026-10-13)

      binary: rustc

      commit-hash: 58cf8f9c35b355199727627d07ae34a4e2846d67

      commit-date: 2026-10-13

      host: x86_64-unknown-linux-gnu

      release: 1.99.1

      LLVM version: 23.1.1

      '
    rustc_release: 1.99.1
    rustflags: null
    target_cpu: null
environment:
  container: none
  hostname: primerose
  os: Linux-6.8.0-147-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-63
    l3_domains:
    - 0-7
    - 8-15
    - 16-23
    - 24-31
    - 32-39
    - 40-47
    - 48-55
    - 56-63
    cpu_boost: enabled
    scaling_governor: schedutil
    scaling_driver: acpi-cpufreq
  env:
    OMP_NUM_THREADS: '1'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    OPENBLAS_NUM_THREADS: '1'
    MATMUL_NUM_THREADS: '1'
    CUDA_HOME: /usr/local/cuda-12.6
    CUDA_PATH: /usr/local/cuda-12.6
    TENFERRO_CUTENSOR_PATH: /usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2
    UV_PROJECT_ENVIRONMENT: /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv
    BENCH_RUNS: '15'
    BENCH_WARMUPS: '3'
timing_policy:
  version: 2
  scope: operation execution and completion synchronization
  setup: fixtures, H2D, runtime/session/handles, plans and compilation outside every
    timer
  cleanup: returned outputs retained until after timer stops
  minimum_untimed_priming_runs: 1
  short_operation_sampling: adaptive batches target 2 ms, at most 4096 operations
    and 256 MiB retained output; output-reuse destinations prepared before timing
  trace_execution: 'tenferro trace excluded: this suite compares immediate per-op
    calls, while trace executes a prepared graph in one caller-held session (run_prepared_in);
    JAX JIT compilation outside timing'
  setup_diagnostics: false
blas:
  implementation: none
python_backends:
  pytorch:
    available: true
    version: 2.12.0+cu130
    file: /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv/lib/python3.14/site-packages/torch/__init__.py
    provider: mkl
    blas_info: mkl
    lapack_info: mkl
    config: "PyTorch built with:\n  - GCC 13.3\n  - C++ Version: 202002\n  - Intel(R)\
      \ oneAPI Math Kernel Library Version 2024.2-Product Build 20240605 for Intel(R)\
      \ 64 architecture applications\n  - Intel(R) MKL-DNN v3.11.2 (Git Hash 03c022d3ffdcee958cfacbe720048e725fdf644c)\n\
      \  - OpenMP 201511 (a.k.a. OpenMP 4.5)\n  - LAPACK is enabled (usually provided\
      \ by MKL)\n  - NNPACK is enabled\n  - CPU capability usage: AVX2\n  - CUDA Runtime\
      \ 13.0\n  - NVCC architecture flags: -gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_100,code=sm_100;-gencode;arch=compute_120,code=sm_120\n\
      \  - CuDNN 92.7  (built against CUDA 12.9)\n    - Built with CuDNN 92.0\n  -\
      \ Magma 2.6.1\n  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, COMMIT_SHA=7661cd9c6b841b62b7f411aa52ec51f05457263b,\
      \ CUDA_FLAGS= -DLIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS -Xfatbin -compress-all\
      \ -DONNX_NAMESPACE=onnx_torch -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80\
      \ -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode\
      \ arch=compute_100,code=sm_100 -gencode arch=compute_120,code=sm_120 -Xcudafe\
      \ --diag_suppress=cc_clobber_ignored,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl\
      \ --expt-relaxed-constexpr --expt-extended-lambda -Xfatbin -compress-all --threads\
      \ 2 -compress-mode=size -Wno-deprecated-gpu-targets --expt-extended-lambda -DCUB_WRAPPED_NAMESPACE=at_cuda_detail\
      \ -DDISABLE_CUSPARSE_DEPRECATED -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__\
      \ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__\
      \ -DC10_NODEPRECATED, CUDA_VERSION=13.0, CUDNN_VERSION=9.20.0, CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++,\
      \ CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO\
      \ -DLIBKINETO_NOROCTRACER -DLIBKINETO_NOXPUPTI=ON -DUSE_FBGEMM -DUSE_MSLK -DUSE_PYTORCH_QNNPACK\
      \ -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED\
      \ -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct\
      \ -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas\
      \ -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow\
      \ -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized\
      \ -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference\
      \ -Wno-error=dangling-reference -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1,\
      \ PERF_WITH_AVX2=1, TORCH_VERSION=2.12.0, USE_CUDA=ON, USE_CUDNN=ON, USE_CUSPARSELT=1,\
      \ USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF,\
      \ USE_NCCL=1, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF,\
      \ USE_XCCL=OFF, USE_XPU=OFF, \n"
    library: /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv/lib/python3.14/site-packages/torch/lib/libtorch_cpu.so
    linked_libraries:
    - /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv/lib/python3.14/site-packages/torch/lib/libgomp.so.1
  jax:
    available: true
    version: 0.10.1
    jaxlib_version: 0.10.1
    backend: gpu
    file: /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv/lib/python3.14/site-packages/jaxlib/__init__.py
    provider: xla_gpu
    dot_backend: xla_gpu
    lapack_provider: none_detected
    library: /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv/lib/python3.14/site-packages/jaxlib/_ifrt_proxy.so
    linked_libraries: []
julia:
  available: false
  reason: julia not found on PATH
cuda:
  device_ordinal: 0
  device_name: NVIDIA A100 80GB PCIe
  device_uuid: GPU-530977e1-4968-9283-4129-9fbec3e66542
  device_memory_bytes: 85899345920
  driver_version: 580.178.04
  runtime_version: '12.6'
  cuda_version: '13.0'
  toolkit_version: '12.6'
  toolkit_root: /usr/local/cuda-12.6
  cutensor_version: 2.8.0
  cusolver_version: 11.7.1
  cudnn_version: '92700'
  artifact_cache:
    path: null
    regime: 'off'
gpu_memory_calibration:
  kind: device_copy
  method: torch.Tensor.copy_ between two contiguous device buffers, one copy per sample,
    device synchronized before and after the clock
  device_ordinal: 0
  device_name: NVIDIA A100 80GB PCIe
  buffer_bytes: 67108864
  traffic_bytes_per_copy: 134217728
  traffic_model: one read plus one write of buffer_bytes per copy
  warmups: 3
  sample_ms:
  - 0.095102
  - 0.094262
  - 0.096482
  - 0.094612
  - 0.094972
  - 0.096682
  - 0.095451
  - 0.094141
  - 0.096002
  - 0.095592
  - 0.094922
  - 0.094162
  - 0.097302
  - 0.094102
  - 0.094322
  median_ms: 0.094972
  bytes_per_second: 1413234721812.7449
  torch_version: 2.12.0+cu130
  cuda_version: '13.0'
  timestamp: '2026-11-05T08:03:20.201616+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-530977e1-4968-9283-4129-9fbec3e66542
  cuda_visible_devices: GPU-530977e1-4968-9283-4129-9fbec3e66542
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh reuse_contraction
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 56
  selected_cases: 56
  sha256:
    .devcontainer/cuda/devcontainer.json: 0bf1465bbd1a4c9f896d5320838f2d5a488632a59af7f9b59cc050e95e0b8e09
    .devcontainer/cuda/Dockerfile: 578d9815e5145aa1651ac7e93833468c345ee8bf5e95bfa7717551580ae9feca
    src/bin/benchmark_gpu_public_api.rs: 4e8f3c519e9210dbc8725dac34da23e543bf462d10c4a202c61e80b59ed108dd
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 9c3d7916a3281d2dfd1caa68551dc9c81c79dab6b7525324339e5c4816581e6d
    src/bin/benchmark_gpu_public_api/multi.rs: 949eadd6272c17ba4e380a228547a03608d328a5ef852eb95294731c288f24d9
    src/bin/benchmark_gpu_public_api/plan.rs: 70fce8a6e9ed7518779a51a9d8692e010fa1e470e95eee0ce5276f595737fde8
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: a5a4aa0edb0a1158f6669b642f4bc25a7079314fc2a711ef3ef57cc19c0a35b2
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: c9d61b1e99bda517a59a6e34f625fd4d651bbbb19f6dc06ba4d0d1c0d50dfd6a
    src/bin/benchmark_gpu_public_api/state.rs: a8bbf8985bbc58c435e3c4dda7135161a141a90a10fa50d4f95128397ce949cb
    src/bin/benchmark_gpu_public_api/structural.rs: 8b61ac01b58edc090eafbc124ec135d5efeb4979631a939656ba78dd6e5c042a
    /home/shinaoka/work/wallclock/scripts/benchmark_gpu_public_api.py: 1919e33cc5f2d0208901837f80cb7c0600426462f998be6dd933062d9f6bb345
    /home/shinaoka/work/wallclock/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 31587e7a83440d99417be00110c4dd12c5cc1143e620bf24b1c03fe5f76bfb5e
    data/results/nvidia-gpu/gpu/reuse_contraction/20261105_0820/fixtures/manifest.json: 706a6a5d725bf440a03748d8ba95eeff194de50a81b8547ec97f56312b6629ef
    target/release/benchmark_gpu_public_api: efd5ab70e2c4de27c458d808d308b8bf3415012551220c291451a2591d08880f
  reference_numpy_version: 2.4.2
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| add_into_f64_n1048576 | tenferro-cuda-direct | add_into_f64_n1048576 untimed 530.3 ms is 5070% of the 10.5 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | add_read_into_f64_n1048576 untimed 316.6 ms is 3028% of the 10.5 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | tenferro-cuda-direct | sub_into_f64_n1048576 untimed 312.8 ms is 3013% of the 10.4 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | sub_read_into_f64_n1048576 untimed 312.8 ms is 3003% of the 10.4 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | tenferro-cuda-direct | mul_into_f64_n1048576 untimed 307.5 ms is 2964% of the 10.4 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | mul_read_into_f64_n1048576 untimed 298.9 ms is 2892% of the 10.3 ms timed region (budget 20%) |
| div_into_f64_n1048576 | tenferro-cuda-direct | div_into_f64_n1048576 untimed 299.8 ms is 2902% of the 10.3 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | div_read_into_f64_n1048576 untimed 298.3 ms is 2896% of the 10.3 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | tenferro-cuda-direct | neg_into_f64_n1048576 untimed 1261.6 ms is 14974% of the 8.4 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | neg_read_into_f64_n1048576 untimed 300.6 ms is 3555% of the 8.5 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | tenferro-cuda-direct | conj_into_c64_n1048576 untimed 356.9 ms is 4442% of the 8.0 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | conj_read_into_c64_n1048576 untimed 357.4 ms is 4435% of the 8.1 ms timed region (budget 20%) |
| tensordot_f64_n262144 | tenferro-cuda-direct | tensordot_f64_n262144 untimed 441.7 ms is 1100% of the 40.1 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | tenferro-cuda-direct | vdot_read_f64_n262144 untimed 53906.5 ms is 127835% of the 42.2 ms timed region (budget 20%) |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | axpby_read_into_accum_f64_n1048576 untimed 262.7 ms is 1741% of the 15.1 ms timed region (budget 20%) |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | dot_general_read_into_f64_s256 untimed 249.9 ms is 744% of the 33.6 ms timed region (budget 20%) |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | dot_general_read_into_accum_f64_s256 untimed 277.0 ms is 738% of the 37.5 ms timed region (budget 20%) |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | dot_general_read_into_accum_cached_f64_s256 untimed 293.3 ms is 744% of the 39.4 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | add_strided_read_f64_n1048576 untimed 61.3 ms is 120% of the 51.2 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | mul_strided_read_f64_n1048576 untimed 60.8 ms is 118% of the 51.7 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | neg_strided_read_f64_n1048576 untimed 46.6 ms is 150% of the 31.0 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | exp_strided_read_f64_n1048576 untimed 48.1 ms is 159% of the 30.3 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | tanh_strided_read_f64_n1048576 untimed 50.2 ms is 160% of the 31.4 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_sum_strided_read_f64_n1048576 untimed 19.0 ms is 39% of the 48.6 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | conj_strided_read_c64_n1048576 untimed 77.9 ms is 354% of the 22.0 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | dot_general_batch_f64_s256 untimed 27.4 ms is 54% of the 50.8 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | dot_conj_l0r0_c64_n262144 untimed 18.0 ms is 52% of the 34.8 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | tenferro-cuda-direct | einsum_plan_owned_f64_s256 untimed 9.4 ms is 21% of the 45.4 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | tenferro-cuda-direct | einsum_plan_read_f64_s256 untimed 9.0 ms is 20% of the 44.9 ms timed region (budget 20%) |
| einsum_plan_into_f64_s256 | tenferro-cuda-direct | einsum_plan_into_f64_s256 untimed 407.8 ms is 1310% of the 31.1 ms timed region (budget 20%) |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-direct | einsum_plan_read_into_f64_s256 untimed 396.2 ms is 1287% of the 30.8 ms timed region (budget 20%) |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-direct | einsum_plan_read_into_accum_f64_s256 untimed 417.7 ms is 1208% of the 34.6 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_into_f64_s256 untimed 392.0 ms is 1343% of the 29.2 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_into_accum_f64_s256 untimed 395.6 ms is 1323% of the 29.9 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | tenferro-cuda-direct | sub_strided_read_f64_n1048576 untimed 58.7 ms is 116% of the 50.5 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | tenferro-cuda-direct | div_strided_read_f64_n1048576 untimed 58.3 ms is 116% of the 50.1 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | tenferro-cuda-direct | rem_strided_read_f64_n1048576 untimed 54.5 ms is 176% of the 30.9 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | tenferro-cuda-direct | pow_strided_read_f64_n1048576 untimed 57.3 ms is 181% of the 31.7 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-direct | maximum_strided_read_f64_n1048576 untimed 58.1 ms is 102% of the 56.7 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-direct | minimum_strided_read_f64_n1048576 untimed 58.4 ms is 103% of the 56.9 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | tenferro-cuda-direct | abs_strided_read_f64_n1048576 untimed 43.4 ms is 146% of the 29.8 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | tenferro-cuda-direct | sign_strided_read_f64_n1048576 untimed 45.0 ms is 151% of the 29.9 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-direct | expm1_strided_read_f64_n1048576 untimed 46.5 ms is 151% of the 30.8 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | tenferro-cuda-direct | log_strided_read_f64_n1048576 untimed 46.0 ms is 144% of the 31.9 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-direct | log1p_strided_read_f64_n1048576 untimed 55.3 ms is 173% of the 31.9 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | tenferro-cuda-direct | erf_strided_read_f64_n1048576 untimed 59.2 ms is 179% of the 33.0 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | tenferro-cuda-direct | sin_strided_read_f64_n1048576 untimed 49.3 ms is 157% of the 31.4 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | tenferro-cuda-direct | cos_strided_read_f64_n1048576 untimed 47.7 ms is 150% of the 31.7 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | sqrt_strided_read_f64_n1048576 untimed 46.8 ms is 155% of the 30.1 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | rsqrt_strided_read_f64_n1048576 untimed 51.6 ms is 172% of the 30.1 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_prod_strided_read_f64_n1048576 untimed 19.0 ms is 40% of the 47.5 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_max_strided_read_f64_n1048576 untimed 18.1 ms is 38% of the 47.9 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_min_strided_read_f64_n1048576 untimed 18.5 ms is 38% of the 48.0 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_sum_squares_strided_read_f64_n1048576 untimed 18.1 ms is 38% of the 47.5 ms timed region (budget 20%) |
| tensordot_f64_n262144 | tenferro-cuda-eager | tensordot_f64_n262144 untimed 46868.8 ms is 107192% of the 43.7 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | dot_general_batch_f64_s256 untimed 15.5 ms is 28% of the 55.5 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | dot_conj_l0r0_c64_n262144 untimed 14.7 ms is 42% of the 35.1 ms timed region (budget 20%) |
| add_into_f64_n1048576 | pytorch-cuda | add_into_f64_n1048576 untimed 1332.8 ms is 21988% of the 6.1 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | pytorch-cuda | add_read_into_f64_n1048576 untimed 1210.1 ms is 19974% of the 6.1 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | pytorch-cuda | sub_into_f64_n1048576 untimed 1250.8 ms is 20576% of the 6.1 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | pytorch-cuda | sub_read_into_f64_n1048576 untimed 1218.2 ms is 20052% of the 6.1 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | pytorch-cuda | mul_into_f64_n1048576 untimed 1235.5 ms is 20055% of the 6.2 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | pytorch-cuda | mul_read_into_f64_n1048576 untimed 1211.1 ms is 19748% of the 6.1 ms timed region (budget 20%) |
| div_into_f64_n1048576 | pytorch-cuda | div_into_f64_n1048576 untimed 1241.3 ms is 19351% of the 6.4 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | pytorch-cuda | div_read_into_f64_n1048576 untimed 1209.3 ms is 18940% of the 6.4 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | pytorch-cuda | neg_into_f64_n1048576 untimed 1236.4 ms is 32879% of the 3.8 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | pytorch-cuda | neg_read_into_f64_n1048576 untimed 1230.4 ms is 32657% of the 3.8 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | pytorch-cuda | conj_into_c64_n1048576 untimed 1253.6 ms is 20651% of the 6.1 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | pytorch-cuda | conj_read_into_c64_n1048576 untimed 1278.6 ms is 21034% of the 6.1 ms timed region (budget 20%) |
| tensordot_f64_n262144 | pytorch-cuda | tensordot_f64_n262144 untimed 349.4 ms is 717% of the 48.7 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | pytorch-cuda | vdot_read_f64_n262144 untimed 9.1 ms is 24% of the 37.7 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | pytorch-cuda | norm_squared_read_f64_n262144 untimed 7.8 ms is 21% of the 36.6 ms timed region (budget 20%) |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | axpby_read_into_accum_f64_n1048576 untimed 1243.9 ms is 15858% of the 7.8 ms timed region (budget 20%) |
| dot_general_read_into_f64_s256 | pytorch-cuda | dot_general_read_into_f64_s256 untimed 97.7 ms is 259% of the 37.7 ms timed region (budget 20%) |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | dot_general_read_into_accum_f64_s256 untimed 119.6 ms is 283% of the 42.2 ms timed region (budget 20%) |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | dot_general_read_into_accum_cached_f64_s256 untimed 108.7 ms is 256% of the 42.5 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | pytorch-cuda | add_strided_read_f64_n1048576 untimed 1260.8 ms is 20769% of the 6.1 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | pytorch-cuda | mul_strided_read_f64_n1048576 untimed 1257.7 ms is 20514% of the 6.1 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | pytorch-cuda | neg_strided_read_f64_n1048576 untimed 1231.4 ms is 32423% of the 3.8 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | pytorch-cuda | exp_strided_read_f64_n1048576 untimed 1197.2 ms is 29866% of the 4.0 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | tanh_strided_read_f64_n1048576 untimed 1235.4 ms is 31346% of the 3.9 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_strided_read_f64_n1048576 untimed 38.3 ms is 75% of the 51.1 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | pytorch-cuda | conj_strided_read_c64_n1048576 untimed 1312.9 ms is 21520% of the 6.1 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | pytorch-cuda | dot_general_batch_f64_s256 untimed 259.2 ms is 642% of the 40.3 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | dot_conj_l0r0_c64_n262144 untimed 11.1 ms is 22% of the 49.7 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | pytorch-cuda | einsum_plan_owned_f64_s256 untimed 87.1 ms is 222% of the 39.2 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | pytorch-cuda | einsum_plan_read_f64_s256 untimed 86.8 ms is 222% of the 39.1 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | einsum_plan_strided_read_f64_s256 untimed 87.4 ms is 220% of the 39.7 ms timed region (budget 20%) |
| einsum_plan_into_f64_s256 | pytorch-cuda | einsum_plan_into_f64_s256 untimed 83.5 ms is 248% of the 33.7 ms timed region (budget 20%) |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | einsum_plan_read_into_f64_s256 untimed 94.3 ms is 284% of the 33.2 ms timed region (budget 20%) |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_read_into_accum_f64_s256 untimed 112.6 ms is 269% of the 41.9 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_f64_s256 untimed 88.4 ms is 259% of the 34.2 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_accum_f64_s256 untimed 113.0 ms is 264% of the 42.7 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | pytorch-cuda | sub_strided_read_f64_n1048576 untimed 1235.4 ms is 20128% of the 6.1 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | pytorch-cuda | div_strided_read_f64_n1048576 untimed 1257.7 ms is 19744% of the 6.4 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | pytorch-cuda | rem_strided_read_f64_n1048576 untimed 1264.1 ms is 21447% of the 5.9 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | pytorch-cuda | pow_strided_read_f64_n1048576 untimed 1276.8 ms is 9149% of the 14.0 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | maximum_strided_read_f64_n1048576 untimed 1271.4 ms is 21755% of the 5.8 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | minimum_strided_read_f64_n1048576 untimed 1232.4 ms is 21151% of the 5.8 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | pytorch-cuda | abs_strided_read_f64_n1048576 untimed 1219.7 ms is 32213% of the 3.8 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | pytorch-cuda | sign_strided_read_f64_n1048576 untimed 1203.8 ms is 32156% of the 3.7 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | expm1_strided_read_f64_n1048576 untimed 1205.8 ms is 25359% of the 4.8 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | pytorch-cuda | log_strided_read_f64_n1048576 untimed 1217.5 ms is 20094% of the 6.1 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | log1p_strided_read_f64_n1048576 untimed 1215.3 ms is 19399% of the 6.3 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | pytorch-cuda | erf_strided_read_f64_n1048576 untimed 1214.9 ms is 17635% of the 6.9 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | pytorch-cuda | sin_strided_read_f64_n1048576 untimed 1205.3 ms is 24904% of the 4.8 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | pytorch-cuda | cos_strided_read_f64_n1048576 untimed 1215.0 ms is 24976% of the 4.9 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | sqrt_strided_read_f64_n1048576 untimed 1204.3 ms is 31816% of the 3.8 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | rsqrt_strided_read_f64_n1048576 untimed 1240.6 ms is 32760% of the 3.8 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | reduce_prod_strided_read_f64_n1048576 untimed 27.4 ms is 54% of the 51.2 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | reduce_max_strided_read_f64_n1048576 untimed 31.7 ms is 62% of the 51.4 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | reduce_min_strided_read_f64_n1048576 untimed 32.0 ms is 62% of the 51.6 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_squares_strided_read_f64_n1048576 untimed 25.6 ms is 62% of the 41.6 ms timed region (budget 20%) |
| tensordot_f64_n262144 | jax-cuda | tensordot_f64_n262144 untimed 1144.5 ms is 2486% of the 46.0 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | jax-cuda | vdot_read_f64_n262144 untimed 244.9 ms is 711% of the 34.5 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | jax-cuda | norm_squared_read_f64_n262144 untimed 7.6 ms is 25% of the 30.0 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | jax-cuda | add_strided_read_f64_n1048576 untimed 1609.3 ms is 5135% of the 31.3 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | jax-cuda | mul_strided_read_f64_n1048576 untimed 1302.6 ms is 4529% of the 28.8 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | jax-cuda | neg_strided_read_f64_n1048576 untimed 1264.7 ms is 4512% of the 28.0 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | jax-cuda | exp_strided_read_f64_n1048576 untimed 1283.2 ms is 5677% of the 22.6 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | jax-cuda | tanh_strided_read_f64_n1048576 untimed 1297.8 ms is 5272% of the 24.6 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | reduce_sum_strided_read_f64_n1048576 untimed 201.1 ms is 686% of the 29.3 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | jax-cuda | conj_strided_read_c64_n1048576 untimed 1522.1 ms is 12310% of the 12.4 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | jax-cuda | dot_general_batch_f64_s256 untimed 936.3 ms is 2188% of the 42.8 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | jax-cuda | dot_conj_l0r0_c64_n262144 untimed 129.9 ms is 240% of the 54.1 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | jax-cuda | einsum_plan_owned_f64_s256 untimed 649.5 ms is 1242% of the 52.3 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | jax-cuda | einsum_plan_read_f64_s256 untimed 93.5 ms is 178% of the 52.7 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | jax-cuda | einsum_plan_strided_read_f64_s256 untimed 404.4 ms is 769% of the 52.6 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | jax-cuda | sub_strided_read_f64_n1048576 untimed 1387.9 ms is 4599% of the 30.2 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | jax-cuda | div_strided_read_f64_n1048576 untimed 1343.9 ms is 5114% of the 26.3 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | jax-cuda | rem_strided_read_f64_n1048576 untimed 1319.5 ms is 5222% of the 25.3 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | jax-cuda | pow_strided_read_f64_n1048576 untimed 1292.0 ms is 4266% of the 30.3 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | jax-cuda | maximum_strided_read_f64_n1048576 untimed 1301.2 ms is 4518% of the 28.8 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | jax-cuda | minimum_strided_read_f64_n1048576 untimed 1268.2 ms is 4414% of the 28.7 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | jax-cuda | abs_strided_read_f64_n1048576 untimed 1295.3 ms is 5303% of the 24.4 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | jax-cuda | sign_strided_read_f64_n1048576 untimed 1360.6 ms is 5410% of the 25.2 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | jax-cuda | expm1_strided_read_f64_n1048576 untimed 1415.9 ms is 5551% of the 25.5 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | jax-cuda | log_strided_read_f64_n1048576 untimed 1383.3 ms is 5381% of the 25.7 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | jax-cuda | log1p_strided_read_f64_n1048576 untimed 1323.0 ms is 5474% of the 24.2 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | jax-cuda | erf_strided_read_f64_n1048576 untimed 1310.8 ms is 3906% of the 33.6 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | jax-cuda | sin_strided_read_f64_n1048576 untimed 1322.1 ms is 5362% of the 24.7 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | jax-cuda | cos_strided_read_f64_n1048576 untimed 1316.6 ms is 5583% of the 23.6 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | jax-cuda | sqrt_strided_read_f64_n1048576 untimed 1332.6 ms is 5848% of the 22.8 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | rsqrt_strided_read_f64_n1048576 untimed 1324.6 ms is 3758% of the 35.2 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | reduce_prod_strided_read_f64_n1048576 untimed 195.4 ms is 391% of the 50.0 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | reduce_max_strided_read_f64_n1048576 untimed 276.6 ms is 558% of the 49.5 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | reduce_min_strided_read_f64_n1048576 untimed 117.5 ms is 260% of the 45.3 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | reduce_sum_squares_strided_read_f64_n1048576 untimed 250.2 ms is 493% of the 50.8 ms timed region (budget 20%) |