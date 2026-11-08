# gpu/reuse_contraction: CUDA public API benchmark

Measured cells: **59**. Unsupported cells: **53**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1405.5 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261108_110139/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**3** of 56) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Transposed borrowed-read cases use prepared native CUDA views in tenferro and PyTorch. JAX compares the same logical values in its native representation; its columns do not claim an equivalent public strided-view contract. View creation and input conversion are outside timing.

Generalized batch dot preserves native output order: tenferro [m,n,b], PyTorch/JAX [b,m,n]. Input permutation views and dimension-number configurations are prepared before timing. Output axes are normalized only on downloaded CPU arrays for verification; no timed output transpose or copy is added.

Prepared einsum uses the public ConcreteEinsumPlan for ik,kj->ij, built once outside execution and sampling. Native matrix multiplication/reuse/accumulation references match the selected mathematical and output-ownership contracts; they do not assert an equivalent prepared-plan interface. Transposed input views/values, read/write handles and accumulation seed restoration are untimed.

| Case | tenferro-cuda-eager | pytorch-cuda |
| --- | ---: | ---: |
| add_into_f64_n1048576 | unsupported | 12.028 (0.179) |
| add_read_into_f64_n1048576 | unsupported | 11.710 (0.182) |
| sub_into_f64_n1048576 | unsupported | 11.998 (0.080) |
| sub_read_into_f64_n1048576 | unsupported | 12.201 (0.138) |
| mul_into_f64_n1048576 | unsupported | 12.124 (0.140) |
| mul_read_into_f64_n1048576 | unsupported | 12.385 (0.209) |
| div_into_f64_n1048576 | unsupported | 13.385 (0.179) |
| div_read_into_f64_n1048576 | unsupported | 13.281 (0.148) |
| neg_into_f64_n1048576 | unsupported | 7.610 (0.044) |
| neg_read_into_f64_n1048576 | unsupported | 7.724 (0.044) |
| conj_into_c64_n1048576 | unsupported | 25.033 (0.199) |
| conj_read_into_c64_n1048576 | unsupported | 25.127 (0.304) |
| tensordot_f64_n262144 † | 44.886 (3.165) | 25.042 (0.781) |
| vdot_read_f64_n262144 | unsupported | 9.688 (0.109) |
| norm_squared_read_f64_n262144 | unsupported | 9.150 (0.011) |
| axpby_read_into_accum_f64_n1048576 | unsupported | 16.472 (0.171) |
| dot_general_read_into_f64_s256 | unsupported | 7.520 (0.008) |
| dot_general_read_into_accum_f64_s256 | unsupported | 8.823 (0.011) |
| dot_general_read_into_accum_cached_f64_s256 | unsupported | 8.826 (0.014) |
| add_strided_read_f64_n1048576 | unsupported | 12.410 (0.191) |
| mul_strided_read_f64_n1048576 | unsupported | 12.101 (0.183) |
| neg_strided_read_f64_n1048576 | unsupported | 7.655 (0.120) |
| exp_strided_read_f64_n1048576 | unsupported | 8.087 (0.078) |
| tanh_strided_read_f64_n1048576 | unsupported | 8.058 (0.042) |
| reduce_sum_strided_read_f64_n1048576 | unsupported | 13.507 (0.012) |
| conj_strided_read_c64_n1048576 | unsupported | 25.097 (0.230) |
| dot_general_batch_f64_s256 † | 23.737 (0.174) | 16.512 (0.027) |
| dot_conj_l0r0_c64_n262144 † | 33.414 (0.354) | 12.425 (0.007) |
| einsum_plan_owned_f64_s256 | unsupported | 7.652 (0.003) |
| einsum_plan_read_f64_s256 | unsupported | 7.619 (0.009) |
| einsum_plan_strided_read_f64_s256 | unsupported | 7.500 (0.005) |
| einsum_plan_into_f64_s256 | unsupported | 7.610 (0.006) |
| einsum_plan_read_into_f64_s256 | unsupported | 7.591 (0.003) |
| einsum_plan_read_into_accum_f64_s256 | unsupported | 8.781 (0.020) |
| einsum_plan_strided_read_into_f64_s256 | unsupported | 7.506 (0.006) |
| einsum_plan_strided_read_into_accum_f64_s256 | unsupported | 8.657 (0.020) |
| sub_strided_read_f64_n1048576 | unsupported | 12.390 (0.129) |
| div_strided_read_f64_n1048576 | unsupported | 13.288 (0.168) |
| rem_strided_read_f64_n1048576 | unsupported | 12.450 (0.179) |
| pow_strided_read_f64_n1048576 | unsupported | 29.100 (0.058) |
| maximum_strided_read_f64_n1048576 | unsupported | 12.498 (0.210) |
| minimum_strided_read_f64_n1048576 | unsupported | 12.222 (0.096) |
| abs_strided_read_f64_n1048576 | unsupported | 7.656 (0.061) |
| sign_strided_read_f64_n1048576 | unsupported | 7.546 (0.046) |
| expm1_strided_read_f64_n1048576 | unsupported | 9.673 (0.088) |
| log_strided_read_f64_n1048576 | unsupported | 12.444 (0.041) |
| log1p_strided_read_f64_n1048576 | unsupported | 12.968 (0.066) |
| erf_strided_read_f64_n1048576 | unsupported | 14.253 (0.045) |
| sin_strided_read_f64_n1048576 | unsupported | 10.838 (0.057) |
| cos_strided_read_f64_n1048576 | unsupported | 10.895 (0.053) |
| sqrt_strided_read_f64_n1048576 | unsupported | 7.697 (0.054) |
| rsqrt_strided_read_f64_n1048576 | unsupported | 7.613 (0.055) |
| reduce_prod_strided_read_f64_n1048576 | unsupported | 13.497 (0.015) |
| reduce_max_strided_read_f64_n1048576 | unsupported | 13.589 (0.015) |
| reduce_min_strided_read_f64_n1048576 | unsupported | 13.647 (0.008) |
| reduce_sum_squares_strided_read_f64_n1048576 | unsupported | 21.149 (0.026) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.245 | 16777216 | 1.559 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.385 | 25165824 | 1.489 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.375 | 25165824 | 1.529 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.397 | 25165824 | 1.443 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.527 | 25165824 | 1.087 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No matching public caller-owned accumulating output API; donation is not output reuse. |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.401 | 33554432 | 0.954 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.402 | 33554432 | 0.950 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_strided_read_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.402 | 33554432 | 0.951 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| cos_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.349 | 16777216 | 1.096 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.428 | 25165824 | 1.338 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.425 | 25165824 | 1.348 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.425 | 25165824 | 1.347 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | 256 | 15 | 3.181 | 8388624 | 0.480 | 1.366e-05 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | 64 | 15 | 2.138 | 8388624 | 0.179 | 1.212e-05 |
| dot_general_batch_f64_s256 | pytorch-cuda | 128 | 15 | 2.114 | 3210240 | 0.138 | 0 |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | 64 | 15 | 1.519 | 3210240 | 0.096 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | 256 | 15 | 2.260 | 2097152 | 0.169 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.259 | 2097152 | 0.169 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | pytorch-cuda | 512 | 15 | 3.850 | 2097152 | 0.198 | 0 |
| dot_general_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| einsum_plan_into_f64_s256 | pytorch-cuda | 498 | 15 | 3.790 | 1070080 | 0.100 | 1.093e-06 |
| einsum_plan_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_owned_f64_s256 | pytorch-cuda | 498 | 15 | 3.811 | 1070080 | 0.099 | 1.093e-06 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_f64_s256 | pytorch-cuda | 498 | 15 | 3.794 | 1070080 | 0.100 | 1.093e-06 |
| einsum_plan_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.248 | 1608704 | 0.130 | 3.261e-07 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | 498 | 15 | 3.780 | 1070080 | 0.100 | 1.093e-06 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | 498 | 15 | 3.735 | 1070080 | 0.102 | 1.093e-06 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.216 | 1608704 | 0.132 | 3.261e-07 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | 498 | 15 | 3.738 | 1070080 | 0.101 | 1.093e-06 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| erf_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.456 | 16777216 | 0.837 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| exp_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.259 | 16777216 | 1.476 | 1.093e-07 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.310 | 16777216 | 1.234 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.415 | 16777216 | 0.920 | 1.106e-07 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.398 | 16777216 | 0.959 | 4.308e-08 |
| log_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.400 | 25165824 | 1.433 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.391 | 25165824 | 1.465 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.388 | 25165824 | 1.477 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.396 | 25165824 | 1.446 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.387 | 25165824 | 1.480 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.244 | 16777216 | 1.569 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.247 | 16777216 | 1.546 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.245 | 16777216 | 1.559 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| norm_squared_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.343 | 2097160 | 0.163 | 4.991e-07 |
| norm_squared_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 2097160 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| pow_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.931 | 25165824 | 0.615 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.479 | 8392704 | 0.439 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.494 | 8392704 | 0.438 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.455 | 8392704 | 0.442 | 4.106e-06 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.707 | 8392704 | 0.282 | 3.309e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.458 | 8392704 | 0.442 | 5.935e-06 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rem_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.398 | 25165824 | 1.438 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.244 | 16777216 | 1.568 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sign_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.241 | 16777216 | 1.582 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sin_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.347 | 16777216 | 1.101 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.246 | 16777216 | 1.551 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.384 | 25165824 | 1.492 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.390 | 25165824 | 1.468 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.396 | 25165824 | 1.445 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.258 | 16777216 | 1.481 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tensordot_f64_n262144 | pytorch-cuda | 128 | 15 | 3.205 | 6291456 | 0.179 | 6.714e-06 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 8 | 15 | 0.359 | 6291456 | 0.100 | 6.714e-06 |
| vdot_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.480 | 4194312 | 0.308 | 2.995e-06 |
| vdot_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 4194312 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261108_110139/gpu_reuse_contraction.yaml
timestamp: '2026-11-08T11:01:39+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: 41f2cdf8bbdf08664bbffe7f2f31efd86ecea146
  dirty: false
resolved_deps:
  strided-perm:
  - 0.8.0
  - 0.12.0
  strided-view:
  - 0.7.1
  threadteam:
  - 0.1.0
  - 0.5.1
host:
  cpu: AMD EPYC 9554 64-Core Processor
  logical_cpus: 252
  power:
    smt:
      active: '0'
      control: notsupported
  numa:
    nodes: 2
    cpulists:
      '0': 0-125
      '1': 126-251
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
  - '64'
  - '65'
  - '66'
  - '67'
  - '68'
  - '69'
  - '70'
  - '71'
  - '72'
  - '73'
  - '74'
  - '75'
  - '76'
  - '77'
  - '78'
  - '79'
  - '80'
  - '81'
  - '82'
  - '83'
  - '84'
  - '85'
  - '86'
  - '87'
  - '88'
  - '89'
  - '90'
  - '91'
  - '92'
  - '93'
  - '94'
  - '95'
  - '96'
  - '97'
  - '98'
  - '99'
  - '100'
  - '101'
  - '102'
  - '103'
  - '104'
  - '105'
  - '106'
  - '107'
  - '108'
  - '109'
  - '110'
  - '111'
  - '112'
  - '113'
  - '114'
  - '115'
  - '116'
  - '117'
  - '118'
  - '119'
  - '120'
  - '121'
  - '122'
  - '123'
  - '124'
  - '125'
  - '126'
  - '127'
  - '128'
  - '129'
  - '130'
  - '131'
  - '132'
  - '133'
  - '134'
  - '135'
  - '136'
  - '137'
  - '138'
  - '139'
  - '140'
  - '141'
  - '142'
  - '143'
  - '144'
  - '145'
  - '146'
  - '147'
  - '148'
  - '149'
  - '150'
  - '151'
  - '152'
  - '153'
  - '154'
  - '155'
  - '156'
  - '157'
  - '158'
  - '159'
  - '160'
  - '161'
  - '162'
  - '163'
  - '164'
  - '165'
  - '166'
  - '167'
  - '168'
  - '169'
  - '170'
  - '171'
  - '172'
  - '173'
  - '174'
  - '175'
  - '176'
  - '177'
  - '178'
  - '179'
  - '180'
  - '181'
  - '182'
  - '183'
  - '184'
  - '185'
  - '186'
  - '187'
  - '188'
  - '189'
  - '190'
  - '191'
  - '192'
  - '193'
  - '194'
  - '195'
  - '196'
  - '197'
  - '198'
  - '199'
  - '200'
  - '201'
  - '202'
  - '203'
  - '204'
  - '205'
  - '206'
  - '207'
  - '208'
  - '209'
  - '210'
  - '211'
  - '212'
  - '213'
  - '214'
  - '215'
  - '216'
  - '217'
  - '218'
  - '219'
  - '220'
  - '221'
  - '222'
  - '223'
  - '224'
  - '225'
  - '226'
  - '227'
  - '228'
  - '229'
  - '230'
  - '231'
  - '232'
  - '233'
  - '234'
  - '235'
  - '236'
  - '237'
  - '238'
  - '239'
  - '240'
  - '241'
  - '242'
  - '243'
  - '244'
  - '245'
  - '246'
  - '247'
  - '248'
  - '249'
  - '250'
  - '251'
  l3_domains:
  - 0-125
  - 126-251
build:
  toolchain:
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: 0db5d65f81dc
  os: Linux-6.8.0-147-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-251
    l3_domains:
    - 0-125
    - 126-251
    cpu_boost: null
    scaling_governor: null
    scaling_driver: null
  env:
    OMP_NUM_THREADS: '1'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    OPENBLAS_NUM_THREADS: '1'
    MATMUL_NUM_THREADS: '1'
    CUDA_HOME: /opt/tenferro-benchmark-ci/cuda
    CUDA_PATH: /opt/tenferro-benchmark-ci/cuda
    TENFERRO_CUTENSOR_PATH: /opt/tenferro-benchmark-ci/cutensor/lib/libcutensor.so.2
    GPU_RUNTIME_SHA256: 7cac8477f65d07d1a9f43b33d9f8fc8f336e03e4302840fc3eea4b16076e41cf
    UV_PROJECT_ENVIRONMENT: /opt/tenferro-benchmark-ci/venv
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
    version: 2.11.0+cu128
    file: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/torch/__init__.py
    provider: mkl
    blas_info: mkl
    lapack_info: mkl
    config: "PyTorch built with:\n  - GCC 13.3\n  - C++ Version: 201703\n  - Intel(R)\
      \ oneAPI Math Kernel Library Version 2024.2-Product Build 20240605 for Intel(R)\
      \ 64 architecture applications\n  - Intel(R) MKL-DNN v3.10.2 (Git Hash f1d471933dc852f956fd05389f9313c7148783d5)\n\
      \  - OpenMP 201511 (a.k.a. OpenMP 4.5)\n  - LAPACK is enabled (usually provided\
      \ by MKL)\n  - NNPACK is enabled\n  - CPU capability usage: AVX512\n  - CUDA\
      \ Runtime 12.8\n  - NVCC architecture flags: -gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_100,code=sm_100;-gencode;arch=compute_120,code=sm_120\n\
      \  - CuDNN 91.9  (built against CUDA 12.9)\n  - Magma 2.6.1\n  - Build settings:\
      \ BLAS_INFO=mkl, BUILD_TYPE=Release, COMMIT_SHA=70d99e998b4955e0049d13a98d77ae1b14db1f45,\
      \ CUDA_VERSION=12.8, CUDNN_VERSION=9.19.0, CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++,\
      \ CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO\
      \ -DLIBKINETO_NOROCTRACER -DLIBKINETO_NOXPUPTI=ON -DUSE_FBGEMM -DUSE_MSLK -DUSE_PYTORCH_QNNPACK\
      \ -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED\
      \ -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct\
      \ -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas\
      \ -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow\
      \ -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized\
      \ -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference\
      \ -Wno-error=dangling-reference -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1,\
      \ PERF_WITH_AVX2=1, TORCH_VERSION=2.11.0, USE_CUDA=ON, USE_CUDNN=ON, USE_CUSPARSELT=1,\
      \ USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF,\
      \ USE_NCCL=1, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF,\
      \ USE_XCCL=OFF, USE_XPU=OFF, \n"
    library: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so
    linked_libraries:
    - /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1
  jax:
    available: true
    version: 0.10.1
    jaxlib_version: 0.10.1
    backend: cpu
    file: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/jaxlib/__init__.py
    provider: xla_cpu
    dot_backend: xla_cpu
    lapack_provider: none_detected
    library: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/jaxlib/_ifrt_proxy.so
    linked_libraries: []
julia:
  available: false
  reason: julia not found on PATH
cuda:
  device_ordinal: 0
  device_name: NVIDIA A100 80GB PCIe
  device_uuid: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
  device_memory_bytes: 85899345920
  driver_version: 595.91.07
  runtime_version: '12.8'
  cuda_version: '13.2'
  toolkit_version: null
  toolkit_root: /opt/tenferro-benchmark-ci/cuda
  cutensor_version: 2.2.0
  cusolver_version: 11.7.3
  cudnn_version: '91900'
  artifact_cache:
    path: /root/.cache/tenferro-benchmark/cuda-artifacts/12.8.93/sm_80
    regime: warm
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
  - 0.094782
  - 0.09244
  - 0.096044
  - 0.096416
  - 0.09342
  - 0.095874
  - 0.095054
  - 0.09347
  - 0.096294
  - 0.095495
  - 0.09346
  - 0.095583
  - 0.095864
  - 0.098929
  - 0.093972
  median_ms: 0.095495
  bytes_per_second: 1405494821718.4146
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-08T11:09:22.157595+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
  cuda_visible_devices: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
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
    src/bin/benchmark_gpu_public_api.rs: 28b921283d595842fc599670df7fadfa2b226573a287cb9fd90799300ca81e6c
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 9c3d7916a3281d2dfd1caa68551dc9c81c79dab6b7525324339e5c4816581e6d
    src/bin/benchmark_gpu_public_api/multi.rs: 949eadd6272c17ba4e380a228547a03608d328a5ef852eb95294731c288f24d9
    src/bin/benchmark_gpu_public_api/plan.rs: 70fce8a6e9ed7518779a51a9d8692e010fa1e470e95eee0ce5276f595737fde8
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: a5a4aa0edb0a1158f6669b642f4bc25a7079314fc2a711ef3ef57cc19c0a35b2
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: c9d61b1e99bda517a59a6e34f625fd4d651bbbb19f6dc06ba4d0d1c0d50dfd6a
    src/bin/benchmark_gpu_public_api/state.rs: a8bbf8985bbc58c435e3c4dda7135161a141a90a10fa50d4f95128397ce949cb
    src/bin/benchmark_gpu_public_api/structural.rs: 8b61ac01b58edc090eafbc124ec135d5efeb4979631a939656ba78dd6e5c042a
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 32aa2ab45453502fc16479f429191aa24f867c8d5ad5f59d5a2752a3da413ee5
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 257228952e48c61fa1714d0cb2d6fdfdb9b0cf3898f59307ba9e72e513aa1b50
    data/results/nvidia-gpu/gpu/reuse_contraction/20261108_110139/fixtures/manifest.json: 706a6a5d725bf440a03748d8ba95eeff194de50a81b8547ec97f56312b6629ef
    target/release/benchmark_gpu_public_api: 62f92e0c72d6fb46531ee8106f112567dfd32ddb482255a7cc41ba6819929786
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T11:09:35Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T11:09:27Z'
  compile:
    case_wall_seconds: 7.185
    compiling_processes: 0
    modules_compiled: 0
    compile_time_ms: null
    compile_time_scope: 'not measured: no compile interval was recorded; `compile_modules`
      counts the NVRTC modules and the untimed wall is the case wall'
    untimed_wall_seconds: 0.0
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| tensordot_f64_n262144 | tenferro-cuda-eager | tensordot_f64_n262144 untimed 2266.1 ms is 41374% of the 5.5 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | dot_general_batch_f64_s256 untimed 12.1 ms is 53% of the 22.8 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | dot_conj_l0r0_c64_n262144 untimed 14.9 ms is 46% of the 32.2 ms timed region (budget 20%) |
| add_into_f64_n1048576 | pytorch-cuda | add_into_f64_n1048576 untimed 182.8 ms is 3172% of the 5.8 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | pytorch-cuda | add_read_into_f64_n1048576 untimed 64.2 ms is 1142% of the 5.6 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | pytorch-cuda | sub_into_f64_n1048576 untimed 55.1 ms is 958% of the 5.8 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | pytorch-cuda | sub_read_into_f64_n1048576 untimed 53.2 ms is 910% of the 5.8 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | pytorch-cuda | mul_into_f64_n1048576 untimed 59.5 ms is 1027% of the 5.8 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | pytorch-cuda | mul_read_into_f64_n1048576 untimed 50.8 ms is 859% of the 5.9 ms timed region (budget 20%) |
| div_into_f64_n1048576 | pytorch-cuda | div_into_f64_n1048576 untimed 64.0 ms is 998% of the 6.4 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | pytorch-cuda | div_read_into_f64_n1048576 untimed 53.0 ms is 832% of the 6.4 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | pytorch-cuda | neg_into_f64_n1048576 untimed 66.1 ms is 1807% of the 3.7 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | pytorch-cuda | neg_read_into_f64_n1048576 untimed 62.5 ms is 1687% of the 3.7 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | pytorch-cuda | conj_into_c64_n1048576 untimed 77.8 ms is 1293% of the 6.0 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | pytorch-cuda | conj_read_into_c64_n1048576 untimed 76.8 ms is 1276% of the 6.0 ms timed region (budget 20%) |
| tensordot_f64_n262144 | pytorch-cuda | tensordot_f64_n262144 untimed 63.1 ms is 130% of the 48.6 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | pytorch-cuda | vdot_read_f64_n262144 untimed 11.7 ms is 31% of the 37.1 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | pytorch-cuda | norm_squared_read_f64_n262144 untimed 9.5 ms is 27% of the 35.1 ms timed region (budget 20%) |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | axpby_read_into_accum_f64_n1048576 untimed 66.6 ms is 845% of the 7.9 ms timed region (budget 20%) |
| dot_general_read_into_f64_s256 | pytorch-cuda | dot_general_read_into_f64_s256 untimed 34.5 ms is 60% of the 57.8 ms timed region (budget 20%) |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | dot_general_read_into_accum_f64_s256 untimed 161.3 ms is 476% of the 33.9 ms timed region (budget 20%) |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | dot_general_read_into_accum_cached_f64_s256 untimed 34.6 ms is 102% of the 33.9 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | pytorch-cuda | add_strided_read_f64_n1048576 untimed 73.5 ms is 1233% of the 6.0 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | pytorch-cuda | mul_strided_read_f64_n1048576 untimed 72.8 ms is 594% of the 12.3 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | pytorch-cuda | neg_strided_read_f64_n1048576 untimed 58.3 ms is 1588% of the 3.7 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | pytorch-cuda | exp_strided_read_f64_n1048576 untimed 80.1 ms is 2051% of the 3.9 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | tanh_strided_read_f64_n1048576 untimed 63.9 ms is 1648% of the 3.9 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_strided_read_f64_n1048576 untimed 25.0 ms is 48% of the 51.9 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | pytorch-cuda | conj_strided_read_c64_n1048576 untimed 101.9 ms is 1689% of the 6.0 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | pytorch-cuda | dot_general_batch_f64_s256 untimed 32.6 ms is 103% of the 31.7 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | dot_conj_l0r0_c64_n262144 untimed 18.8 ms is 39% of the 47.7 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | pytorch-cuda | einsum_plan_owned_f64_s256 untimed 17.6 ms is 31% of the 57.2 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | pytorch-cuda | einsum_plan_read_f64_s256 untimed 17.2 ms is 30% of the 56.9 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | einsum_plan_strided_read_f64_s256 untimed 16.6 ms is 30% of the 56.1 ms timed region (budget 20%) |
| einsum_plan_into_f64_s256 | pytorch-cuda | einsum_plan_into_f64_s256 untimed 14.9 ms is 26% of the 56.8 ms timed region (budget 20%) |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | einsum_plan_read_into_f64_s256 untimed 25.5 ms is 45% of the 56.7 ms timed region (budget 20%) |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_read_into_accum_f64_s256 untimed 31.0 ms is 92% of the 33.7 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_f64_s256 untimed 16.2 ms is 29% of the 56.1 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_accum_f64_s256 untimed 30.3 ms is 91% of the 33.2 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | pytorch-cuda | sub_strided_read_f64_n1048576 untimed 67.1 ms is 1127% of the 6.0 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | pytorch-cuda | div_strided_read_f64_n1048576 untimed 77.9 ms is 1222% of the 6.4 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | pytorch-cuda | rem_strided_read_f64_n1048576 untimed 90.6 ms is 1515% of the 6.0 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | pytorch-cuda | pow_strided_read_f64_n1048576 untimed 90.8 ms is 650% of the 14.0 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | maximum_strided_read_f64_n1048576 untimed 80.0 ms is 1331% of the 6.0 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | minimum_strided_read_f64_n1048576 untimed 57.7 ms is 983% of the 5.9 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | pytorch-cuda | abs_strided_read_f64_n1048576 untimed 44.8 ms is 1219% of the 3.7 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | pytorch-cuda | sign_strided_read_f64_n1048576 untimed 52.0 ms is 1436% of the 3.6 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | expm1_strided_read_f64_n1048576 untimed 50.1 ms is 1077% of the 4.7 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | pytorch-cuda | log_strided_read_f64_n1048576 untimed 55.4 ms is 928% of the 6.0 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | log1p_strided_read_f64_n1048576 untimed 60.3 ms is 969% of the 6.2 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | pytorch-cuda | erf_strided_read_f64_n1048576 untimed 59.3 ms is 867% of the 6.8 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | pytorch-cuda | sin_strided_read_f64_n1048576 untimed 58.1 ms is 1115% of the 5.2 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | pytorch-cuda | cos_strided_read_f64_n1048576 untimed 57.5 ms is 1097% of the 5.2 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | sqrt_strided_read_f64_n1048576 untimed 57.7 ms is 1559% of the 3.7 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | rsqrt_strided_read_f64_n1048576 untimed 63.6 ms is 1735% of the 3.7 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | reduce_prod_strided_read_f64_n1048576 untimed 617.7 ms is 1191% of the 51.8 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | reduce_max_strided_read_f64_n1048576 untimed 19.2 ms is 37% of the 52.2 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | reduce_min_strided_read_f64_n1048576 untimed 23.4 ms is 45% of the 52.4 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_squares_strided_read_f64_n1048576 untimed 19.8 ms is 49% of the 40.6 ms timed region (budget 20%) |