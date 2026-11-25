# gpu/reuse_contraction: CUDA public API benchmark

Measured cells: **162**. Unsupported cells: **62**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1398.2 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261125_070903/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**17** of 56) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Transposed borrowed-read cases use prepared native CUDA views in tenferro and PyTorch. JAX compares the same logical values in its native representation; its columns do not claim an equivalent public strided-view contract. View creation and input conversion are outside timing.

Generalized batch dot preserves native output order: tenferro [m,n,b], PyTorch/JAX [b,m,n]. Input permutation views and dimension-number configurations are prepared before timing. Output axes are normalized only on downloaded CPU arrays for verification; no timed output transpose or copy is added.

Prepared einsum uses the public ConcreteEinsumPlan for ik,kj->ij, built once outside execution and sampling. Native matrix multiplication/reuse/accumulation references match the selected mathematical and output-ownership contracts; they do not assert an equivalent prepared-plan interface. Transposed input views/values, read/write handles and accumulation seed restoration are untimed.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| add_into_f64_n1048576 | 13.140 (0.189) | 13.098 (0.276) | 12.188 (0.148) | unsupported |
| add_read_into_f64_n1048576 | 13.348 (0.337) | 13.104 (0.213) | 11.964 (0.127) | unsupported |
| sub_into_f64_n1048576 | 13.145 (0.201) | 13.155 (0.278) | 11.909 (0.396) | unsupported |
| sub_read_into_f64_n1048576 | 13.110 (0.221) | 13.055 (0.181) | 11.947 (0.158) | unsupported |
| mul_into_f64_n1048576 | 13.196 (0.382) | 13.145 (0.136) | 12.083 (0.186) | unsupported |
| mul_read_into_f64_n1048576 | 13.099 (0.175) | 13.131 (0.160) | 12.068 (0.158) | unsupported |
| div_into_f64_n1048576 | 13.036 (0.072) | 13.018 (0.255) | 13.256 (0.179) | unsupported |
| div_read_into_f64_n1048576 | 13.063 (0.334) | 12.911 (0.232) | 13.243 (0.129) | unsupported |
| neg_into_f64_n1048576 | 8.689 (0.050) | 8.597 (0.036) | 7.758 (0.094) | unsupported |
| neg_read_into_f64_n1048576 | 8.694 (0.048) | 8.667 (0.054) | 7.753 (0.087) | unsupported |
| conj_into_c64_n1048576 | 25.222 (0.207) | 24.985 (0.191) | 25.289 (0.266) | unsupported |
| conj_read_into_c64_n1048576 | 25.203 (0.120) | 25.094 (0.225) | 25.172 (0.226) | unsupported |
| tensordot_f64_n262144 † | 28.176 (0.061) | 28.142 (0.094) | 32.470 (0.825) | 143.422 (3.142) |
| vdot_read_f64_n262144 † | 11.706 (0.236) | unsupported | 10.453 (0.518) | 52.446 (1.057) |
| norm_squared_read_f64_n262144 † | 12.109 (0.296) | unsupported | 10.156 (0.698) | 58.255 (1.259) |
| axpby_read_into_accum_f64_n1048576 † | 159.399 (13.000) | unsupported | 16.579 (0.134) | unsupported |
| dot_general_read_into_f64_s256 | 7.572 (0.044) | unsupported | 11.947 (0.346) | unsupported |
| dot_general_read_into_accum_f64_s256 | 8.829 (0.026) | unsupported | 13.999 (0.342) | unsupported |
| dot_general_read_into_accum_cached_f64_s256 | 8.823 (0.045) | unsupported | 13.528 (0.288) | unsupported |
| add_strided_read_f64_n1048576 † | 34.990 (0.037) | unsupported | 12.285 (0.111) | 58.177 (2.834) |
| mul_strided_read_f64_n1048576 | 34.952 (0.070) | unsupported | 12.127 (0.235) | 58.385 (3.500) |
| neg_strided_read_f64_n1048576 | 19.046 (0.081) | unsupported | 7.856 (0.058) | 48.870 (5.645) |
| exp_strided_read_f64_n1048576 | 19.187 (0.049) | unsupported | 8.199 (0.109) | 40.275 (1.395) |
| tanh_strided_read_f64_n1048576 | 19.183 (0.096) | unsupported | 8.159 (0.064) | 42.529 (1.082) |
| reduce_sum_strided_read_f64_n1048576 † | 36.717 (0.149) | unsupported | 13.422 (0.033) | 45.689 (1.337) |
| conj_strided_read_c64_n1048576 | 30.139 (0.420) | unsupported | 25.423 (0.217) | 43.777 (2.691) |
| dot_general_batch_f64_s256 † | 13.963 (0.194) | 17.049 (0.282) | 23.994 (1.468) | 80.883 (2.801) |
| dot_conj_l0r0_c64_n262144 † | 25.617 (0.049) | 25.419 (0.034) | 13.089 (0.008) | 51.878 (2.851) |
| einsum_plan_owned_f64_s256 † | 14.146 (0.290) | unsupported | 12.849 (0.567) | 46.760 (1.459) |
| einsum_plan_read_f64_s256 † | 14.872 (0.255) | unsupported | 13.293 (0.607) | 47.206 (1.197) |
| einsum_plan_strided_read_f64_s256 † | 37.450 (1.045) | unsupported | 13.498 (0.298) | 53.360 (12.316) |
| einsum_plan_into_f64_s256 | 7.317 (0.112) | unsupported | 10.752 (0.271) | unsupported |
| einsum_plan_read_into_f64_s256 | 7.538 (0.186) | unsupported | 9.898 (1.486) | unsupported |
| einsum_plan_read_into_accum_f64_s256 | 8.475 (0.295) | unsupported | 13.590 (0.822) | unsupported |
| einsum_plan_strided_read_into_f64_s256 † | 25.545 (0.668) | unsupported | 11.009 (0.556) | unsupported |
| einsum_plan_strided_read_into_accum_f64_s256 † | 28.385 (0.327) | unsupported | 13.765 (0.655) | unsupported |
| sub_strided_read_f64_n1048576 | 34.920 (0.110) | unsupported | 12.154 (0.321) | 53.963 (1.310) |
| div_strided_read_f64_n1048576 | 34.968 (0.044) | unsupported | 13.429 (0.168) | 48.579 (1.086) |
| rem_strided_read_f64_n1048576 | 34.913 (0.054) | unsupported | 12.323 (0.270) | 45.331 (1.535) |
| pow_strided_read_f64_n1048576 | 36.436 (0.083) | unsupported | 29.240 (0.074) | 64.122 (2.520) |
| maximum_strided_read_f64_n1048576 | 34.953 (0.088) | unsupported | 12.172 (0.215) | 56.068 (2.808) |
| minimum_strided_read_f64_n1048576 | 34.972 (0.032) | unsupported | 12.135 (0.138) | 55.340 (1.681) |
| abs_strided_read_f64_n1048576 | 19.002 (0.075) | unsupported | 8.027 (0.336) | 44.283 (0.778) |
| sign_strided_read_f64_n1048576 | 19.055 (0.073) | unsupported | 7.710 (0.056) | 43.370 (1.161) |
| expm1_strided_read_f64_n1048576 | 19.284 (0.048) | unsupported | 9.842 (0.091) | 45.334 (2.722) |
| log_strided_read_f64_n1048576 | 19.548 (0.036) | unsupported | 12.456 (0.076) | 45.852 (1.062) |
| log1p_strided_read_f64_n1048576 | 19.530 (0.032) | unsupported | 13.002 (0.075) | 49.127 (2.647) |
| erf_strided_read_f64_n1048576 | 19.781 (0.033) | unsupported | 14.353 (0.050) | 64.170 (0.864) |
| sin_strided_read_f64_n1048576 | 21.188 (0.070) | unsupported | 10.931 (0.044) | 43.696 (0.988) |
| cos_strided_read_f64_n1048576 | 21.150 (0.028) | unsupported | 11.082 (0.052) | 44.811 (0.885) |
| sqrt_strided_read_f64_n1048576 | 19.076 (0.063) | unsupported | 7.832 (0.059) | 44.187 (1.247) |
| rsqrt_strided_read_f64_n1048576 | 19.046 (0.032) | unsupported | 7.773 (0.042) | 67.360 (3.300) |
| reduce_prod_strided_read_f64_n1048576 † | 35.905 (0.045) | unsupported | 13.364 (0.046) | 47.406 (2.393) |
| reduce_max_strided_read_f64_n1048576 † | 36.635 (0.075) | unsupported | 13.489 (0.051) | 49.052 (2.276) |
| reduce_min_strided_read_f64_n1048576 † | 36.630 (0.080) | unsupported | 13.499 (0.056) | 55.099 (2.905) |
| reduce_sum_squares_strided_read_f64_n1048576 † | 36.629 (0.120) | unsupported | 21.249 (0.056) | 923.437 (54.034) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.417 | 16777216 | 0.271 | 0 |
| abs_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.257 | 16777216 | 1.495 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.608 | 16777216 | 0.631 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| add_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.390 | 25165824 | 1.477 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.420 | 25165824 | 1.370 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.419 | 25165824 | 1.374 | 0 |
| add_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.383 | 25165824 | 1.504 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.427 | 25165824 | 1.348 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.419 | 25165824 | 1.374 | 0 |
| add_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 0.931 | 25165824 | 0.309 | 0 |
| add_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.393 | 25165824 | 1.465 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.120 | 25165824 | 0.514 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| axpby_read_into_accum_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.531 | 25165824 | 1.086 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.550 | 25165824 | 0.113 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No matching public caller-owned accumulating output API; donation is not output reuse. |
| conj_into_c64_n1048576 | jax-cuda | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.405 | 33554432 | 0.949 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.404 | 33554432 | 0.951 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.400 | 33554432 | 0.960 | 0 |
| conj_read_into_c64_n1048576 | jax-cuda | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.403 | 33554432 | 0.953 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.403 | 33554432 | 0.952 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.401 | 33554432 | 0.956 | 0 |
| conj_strided_read_c64_n1048576 | jax-cuda | 16 | 15 | 0.700 | 33554432 | 0.548 | 0 |
| conj_strided_read_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.407 | 33554432 | 0.944 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.482 | 33554432 | 0.796 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| cos_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.434 | 16777216 | 0.268 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.355 | 16777216 | 1.083 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.677 | 16777216 | 0.567 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| div_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.424 | 25165824 | 1.358 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.417 | 25165824 | 1.381 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.417 | 25165824 | 1.383 | 0 |
| div_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.424 | 25165824 | 1.359 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.418 | 25165824 | 1.378 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.413 | 25165824 | 1.394 | 0 |
| div_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.555 | 25165824 | 0.370 | 0 |
| div_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.430 | 25165824 | 1.340 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.119 | 25165824 | 0.515 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| dot_conj_l0r0_c64_n262144 | jax-cuda | 64 | 15 | 3.320 | 8388624 | 0.116 | 1.221e-05 |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | 256 | 15 | 3.351 | 8388624 | 0.458 | 1.366e-05 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | 128 | 15 | 3.279 | 8388624 | 0.234 | 1.212e-05 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | 128 | 15 | 3.254 | 8388624 | 0.236 | 1.212e-05 |
| dot_general_batch_f64_s256 | jax-cuda | 32 | 15 | 2.588 | 3210240 | 0.028 | 7.463e-07 |
| dot_general_batch_f64_s256 | pytorch-cuda | 128 | 15 | 3.071 | 3210240 | 0.096 | 7.463e-07 |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | 166 | 15 | 2.318 | 3210240 | 0.164 | 7.463e-07 |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | 128 | 15 | 2.182 | 3210240 | 0.135 | 7.463e-07 |
| dot_general_read_into_accum_cached_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | 256 | 15 | 3.463 | 2097152 | 0.111 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | 256 | 15 | 2.259 | 2097152 | 0.170 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 3.584 | 2097152 | 0.107 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | 256 | 15 | 2.260 | 2097152 | 0.170 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 3.058 | 2097152 | 0.126 | 0 |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | 256 | 15 | 1.938 | 2097152 | 0.198 | 0 |
| dot_general_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| einsum_plan_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.753 | 1070080 | 0.071 | 0 |
| einsum_plan_into_f64_s256 | tenferro-cuda-direct | 256 | 15 | 1.873 | 1070080 | 0.105 | 0 |
| einsum_plan_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_owned_f64_s256 | jax-cuda | 64 | 15 | 2.993 | 1070080 | 0.016 | 0 |
| einsum_plan_owned_f64_s256 | pytorch-cuda | 256 | 15 | 3.289 | 1070080 | 0.060 | 0 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-direct | 128 | 15 | 1.811 | 1070080 | 0.054 | 0 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_f64_s256 | jax-cuda | 64 | 15 | 3.021 | 1070080 | 0.016 | 0 |
| einsum_plan_read_f64_s256 | pytorch-cuda | 256 | 15 | 3.403 | 1070080 | 0.058 | 0 |
| einsum_plan_read_f64_s256 | tenferro-cuda-direct | 128 | 15 | 1.904 | 1070080 | 0.051 | 0 |
| einsum_plan_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_accum_f64_s256 | jax-cuda | — | — | — | 1608704 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 3.479 | 1608704 | 0.085 | 0 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-direct | 256 | 15 | 2.170 | 1608704 | 0.136 | 0 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.534 | 1070080 | 0.077 | 0 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-direct | 256 | 15 | 1.930 | 1070080 | 0.102 | 0 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_f64_s256 | jax-cuda | 64 | 15 | 3.415 | 1070080 | 0.014 | 0 |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | 256 | 15 | 3.455 | 1070080 | 0.057 | 0 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-direct | 64 | 15 | 2.397 | 1070080 | 0.020 | 0 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_accum_f64_s256 | jax-cuda | — | — | — | 1608704 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 3.524 | 1608704 | 0.084 | 0 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-direct | 64 | 15 | 1.817 | 1608704 | 0.041 | 0 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.818 | 1070080 | 0.070 | 0 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-direct | 128 | 15 | 3.270 | 1070080 | 0.030 | 0 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| erf_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.053 | 16777216 | 0.187 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.459 | 16777216 | 0.836 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.633 | 16777216 | 0.607 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| exp_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.289 | 16777216 | 0.298 | 1.095e-07 |
| exp_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.262 | 16777216 | 1.463 | 1.095e-07 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.614 | 16777216 | 0.625 | 1.095e-07 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| expm1_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.451 | 16777216 | 0.265 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.315 | 16777216 | 1.219 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.617 | 16777216 | 0.622 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log1p_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.572 | 16777216 | 0.244 | 1.11e-07 |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.416 | 16777216 | 0.923 | 1.11e-07 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.625 | 16777216 | 0.614 | 1.11e-07 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.467 | 16777216 | 0.262 | 4.308e-08 |
| log_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.399 | 16777216 | 0.963 | 4.308e-08 |
| log_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.626 | 16777216 | 0.614 | 4.308e-08 |
| log_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| maximum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.794 | 25165824 | 0.321 | 0 |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.389 | 25165824 | 1.479 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.118 | 25165824 | 0.515 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| minimum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.771 | 25165824 | 0.325 | 0 |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.388 | 25165824 | 1.483 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.119 | 25165824 | 0.515 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| mul_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.387 | 25165824 | 1.490 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.422 | 25165824 | 1.364 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.421 | 25165824 | 1.369 | 0 |
| mul_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.386 | 25165824 | 1.491 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.419 | 25165824 | 1.374 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.420 | 25165824 | 1.371 | 0 |
| mul_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.868 | 25165824 | 0.308 | 0 |
| mul_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.388 | 25165824 | 1.484 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.118 | 25165824 | 0.515 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| neg_into_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.248 | 16777216 | 1.547 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.278 | 16777216 | 1.381 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.275 | 16777216 | 1.396 | 0 |
| neg_read_into_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.248 | 16777216 | 1.548 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.278 | 16777216 | 1.380 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.277 | 16777216 | 1.384 | 0 |
| neg_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.564 | 16777216 | 0.246 | 0 |
| neg_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.251 | 16777216 | 1.527 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.609 | 16777216 | 0.630 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| norm_squared_read_f64_n262144 | jax-cuda | 64 | 15 | 3.728 | 2097160 | 0.026 | 1.664e-07 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.600 | 2097160 | 0.148 | 1.664e-07 |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | 256 | 15 | 3.100 | 2097160 | 0.124 | 1.664e-07 |
| norm_squared_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 2097160 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| pow_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.052 | 25165824 | 0.281 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.936 | 25165824 | 0.616 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.166 | 25165824 | 0.494 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 3.139 | 8392704 | 0.122 | 0 |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.453 | 8392704 | 0.445 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.345 | 8392704 | 0.164 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 3.526 | 8392704 | 0.109 | 0 |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.456 | 8392704 | 0.445 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.344 | 8392704 | 0.164 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 3.034 | 8392704 | 0.127 | 4.995e-06 |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.421 | 8392704 | 0.449 | 4.106e-06 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.298 | 8392704 | 0.167 | 4.661e-06 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | 4 | 15 | 3.694 | 8392704 | 0.007 | 3.311e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.720 | 8392704 | 0.282 | 3.309e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.344 | 8392704 | 0.164 | 3.312e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 2.924 | 8392704 | 0.131 | 7.159e-06 |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.436 | 8392704 | 0.447 | 5.935e-06 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.350 | 8392704 | 0.163 | 4.724e-06 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rem_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.451 | 25165824 | 0.397 | 0 |
| rem_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.394 | 25165824 | 1.461 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.117 | 25165824 | 0.516 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.156 | 16777216 | 0.178 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.249 | 16777216 | 1.544 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.609 | 16777216 | 0.630 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sign_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.388 | 16777216 | 0.277 | 0 |
| sign_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.247 | 16777216 | 1.556 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.610 | 16777216 | 0.630 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sin_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.398 | 16777216 | 0.275 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.350 | 16777216 | 1.098 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.678 | 16777216 | 0.566 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sqrt_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.414 | 16777216 | 0.272 | 0 |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.251 | 16777216 | 1.532 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.610 | 16777216 | 0.629 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sub_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.381 | 25165824 | 1.511 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.421 | 25165824 | 1.369 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.421 | 25165824 | 1.368 | 0 |
| sub_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.382 | 25165824 | 1.507 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.420 | 25165824 | 1.373 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.418 | 25165824 | 1.379 | 0 |
| sub_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.727 | 25165824 | 0.334 | 0 |
| sub_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.389 | 25165824 | 1.481 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.117 | 25165824 | 0.515 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tanh_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.361 | 16777216 | 0.282 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.261 | 16777216 | 1.471 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.614 | 16777216 | 0.625 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tensordot_f64_n262144 | jax-cuda | 16 | 15 | 2.295 | 6291456 | 0.031 | 6.714e-06 |
| tensordot_f64_n262144 | pytorch-cuda | 128 | 15 | 4.156 | 6291456 | 0.139 | 6.714e-06 |
| tensordot_f64_n262144 | tenferro-cuda-direct | 128 | 15 | 3.607 | 6291456 | 0.160 | 6.714e-06 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 128 | 15 | 3.602 | 6291456 | 0.160 | 6.714e-06 |
| vdot_read_f64_n262144 | jax-cuda | 64 | 15 | 3.357 | 4194312 | 0.057 | 2.995e-06 |
| vdot_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.676 | 4194312 | 0.287 | 2.995e-06 |
| vdot_read_f64_n262144 | tenferro-cuda-direct | 256 | 15 | 2.997 | 4194312 | 0.256 | 2.995e-06 |
| vdot_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 4194312 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: /home/shinaoka/tensor4all/tenferro-benchmark-2701/data/results/nvidia-gpu/gpu/routine_selection/20261125_070903/gpu_reuse_contraction.yaml
timestamp: '2026-11-25T07:09:03+00:00'
tenferro_rs:
  path: /home/shinaoka/tensor4all/tenferro-benchmark-2701/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - cuda
  resolved_path: /home/shinaoka/tensor4all/tenferro-benchmark-2701/extern/tenferro-rs
harness:
  path: /home/shinaoka/tensor4all/tenferro-benchmark-2701
  commit: acdeaf6e4f6845a465352d7b7c3b67c7dbcf005a
  dirty: false
resolved_deps:
  strided-perm:
  - 0.8.0
  - 0.15.1
  strided-view:
  - 0.7.1
  - 0.8.1
  threadteam:
  - 0.1.0
  - 0.6.1
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
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
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
    measured_cpus: 27,29
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
    OMP_THREAD_LIMIT: '1'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    OPENBLAS_NUM_THREADS: '1'
    MATMUL_NUM_THREADS: '1'
    MKL_NUM_THREADS: '1'
    CUDA_HOME: /opt/tenferro-benchmark-ci/cuda
    CUDA_PATH: /opt/tenferro-benchmark-ci/cuda
    TENFERRO_CUTENSOR_PATH: /opt/tenferro-benchmark-ci/cutensor/lib/libcutensor.so.2
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
      \ by MKL)\n  - NNPACK is enabled\n  - CPU capability usage: AVX2\n  - CUDA Runtime\
      \ 12.8\n  - NVCC architecture flags: -gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_100,code=sm_100;-gencode;arch=compute_120,code=sm_120\n\
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
    backend: gpu
    file: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/jaxlib/__init__.py
    provider: xla_gpu
    dot_backend: xla_gpu
    lapack_provider: none_detected
    library: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/jaxlib/_ifrt_proxy.so
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
  runtime_version: '12.8'
  cuda_version: '13.0'
  toolkit_version: null
  toolkit_root: /opt/tenferro-benchmark-ci/cuda
  cutensor_version: 2.2.0
  cusolver_version: 11.7.3
  cudnn_version: '91900'
  artifact_cache:
    path: /home/shinaoka/tensor4all/.t4a-509-gpu-runtime/home/.cache/tenferro-benchmark/cuda-artifacts/12.8.93/sm_80
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
  - 0.096842
  - 0.095471
  - 0.099882
  - 0.098382
  - 0.095011
  - 0.095992
  - 0.096591
  - 0.095062
  - 0.096451
  - 0.096642
  - 0.095841
  - 0.095822
  - 0.096421
  - 0.095972
  - 0.093791
  median_ms: 0.095992
  bytes_per_second: 1398217851487.624
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-25T07:13:21.830395+00:00'
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
    .devcontainer/cuda/devcontainer.json: 402f3763533ea1531d0e400193395302213fd1b151a42baaeaebeafca3acabcd
    .devcontainer/cuda/Dockerfile: 6eff822595fa44f53a6fd9eebe0f4e3cc9e28362d9cfa8cac977a21b10a823f2
    src/bin/benchmark_gpu_public_api.rs: b85ad2743b1594fe0a647253a17fa3bb59800202fa8ed4016d0255f0170bf5b3
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 3d5bedcc3620f75cab2687fdf49210df324021d043b23aec6e169642ceff15ba
    src/bin/benchmark_gpu_public_api/multi.rs: 950cb1b3c1a7cca37665b58e3de074bd906dafb8c84b17a51c27ce797a73afa8
    src/bin/benchmark_gpu_public_api/plan.rs: 016e752c59edefc3ded1405c36325e8680736860cb65954919ea4cf0f3eed061
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: 88038508963fbc13303fb54b3898808e4303131321cabedb9f38255941ead438
    src/bin/benchmark_gpu_public_api/state.rs: e1cc2781ce7392154f32cc19372c9fc41ae29765aedc544c7ae7bf02f873c12b
    src/bin/benchmark_gpu_public_api/structural.rs: 10850458a640d11e79262c215072482ee463bc4ca0604f40d82ae56f345f6564
    src/bin/benchmark_gpu_public_api/reuse.rs: 8934cd12e336764b5ed831f0c92d55f2827142de939a1f33daa5d9ee57b9f482
    /home/shinaoka/tensor4all/tenferro-benchmark-2701/scripts/benchmark_gpu_public_api.py: d1ff68f86496d991bcbc730138fe6e3076001b3f23cc1a94c44b96f6857cb8ed
    /home/shinaoka/tensor4all/tenferro-benchmark-2701/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 7995bcf409258185e07210969a738896f64860b67df251fcd1100d11128a5431
    data/results/nvidia-gpu/gpu/reuse_contraction/20261125_070903/fixtures/manifest.json: d5dfb4abed3ff4b1f486cfb513f451a6683ca3d680c7155b3a3ea242f6ba25b3
    target/release/benchmark_gpu_public_api: 250c27c741c6e6364e4ff58525a6341f7e6a3c3b81e6f429f4c5662a4d8f0f86
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-25T07:14:55Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-25T07:14:29Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-25T07:13:55Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-25T07:14:08Z'
  compile:
    case_wall_seconds: 73.811
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
| add_into_f64_n1048576 | tenferro-cuda-direct | add_into_f64_n1048576 untimed 746.0 ms is 11787% of the 6.3 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | add_read_into_f64_n1048576 untimed 680.0 ms is 10652% of the 6.4 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | tenferro-cuda-direct | sub_into_f64_n1048576 untimed 946.5 ms is 14983% of the 6.3 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | sub_read_into_f64_n1048576 untimed 810.4 ms is 12849% of the 6.3 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | tenferro-cuda-direct | mul_into_f64_n1048576 untimed 820.8 ms is 12879% of the 6.4 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | mul_read_into_f64_n1048576 untimed 558.1 ms is 8844% of the 6.3 ms timed region (budget 20%) |
| div_into_f64_n1048576 | tenferro-cuda-direct | div_into_f64_n1048576 untimed 582.0 ms is 9319% of the 6.2 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | div_read_into_f64_n1048576 untimed 588.8 ms is 9358% of the 6.3 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | tenferro-cuda-direct | neg_into_f64_n1048576 untimed 581.5 ms is 13837% of the 4.2 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | neg_read_into_f64_n1048576 untimed 580.1 ms is 13883% of the 4.2 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | tenferro-cuda-direct | conj_into_c64_n1048576 untimed 594.2 ms is 9816% of the 6.1 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | conj_read_into_c64_n1048576 untimed 601.1 ms is 9955% of the 6.0 ms timed region (budget 20%) |
| tensordot_f64_n262144 | tenferro-cuda-direct | tensordot_f64_n262144 untimed 2458.0 ms is 4538% of the 54.2 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | tenferro-cuda-direct | vdot_read_f64_n262144 untimed 88.7 ms is 197% of the 45.0 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | norm_squared_read_f64_n262144 untimed 25.9 ms is 56% of the 46.5 ms timed region (budget 20%) |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | axpby_read_into_accum_f64_n1048576 untimed 1156.6 ms is 2834% of the 40.8 ms timed region (budget 20%) |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | dot_general_read_into_f64_s256 untimed 422.1 ms is 1440% of the 29.3 ms timed region (budget 20%) |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | dot_general_read_into_accum_f64_s256 untimed 472.7 ms is 1393% of the 33.9 ms timed region (budget 20%) |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | dot_general_read_into_accum_cached_f64_s256 untimed 497.2 ms is 1466% of the 33.9 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | add_strided_read_f64_n1048576 untimed 403.8 ms is 2403% of the 16.8 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | mul_strided_read_f64_n1048576 untimed 383.5 ms is 2285% of the 16.8 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | neg_strided_read_f64_n1048576 untimed 336.5 ms is 3680% of the 9.1 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | exp_strided_read_f64_n1048576 untimed 333.3 ms is 3618% of the 9.2 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | tanh_strided_read_f64_n1048576 untimed 324.2 ms is 3521% of the 9.2 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_sum_strided_read_f64_n1048576 untimed 65.3 ms is 183% of the 35.7 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | conj_strided_read_c64_n1048576 untimed 472.4 ms is 6533% of the 7.2 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | dot_general_batch_f64_s256 untimed 58.7 ms is 168% of the 34.9 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | dot_conj_l0r0_c64_n262144 untimed 55.9 ms is 114% of the 49.2 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | tenferro-cuda-direct | einsum_plan_owned_f64_s256 untimed 26.7 ms is 98% of the 27.3 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | tenferro-cuda-direct | einsum_plan_read_f64_s256 untimed 26.5 ms is 93% of the 28.7 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_f64_s256 untimed 24.8 ms is 69% of the 36.0 ms timed region (budget 20%) |
| einsum_plan_into_f64_s256 | tenferro-cuda-direct | einsum_plan_into_f64_s256 untimed 478.3 ms is 1695% of the 28.2 ms timed region (budget 20%) |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-direct | einsum_plan_read_into_f64_s256 untimed 477.3 ms is 1616% of the 29.5 ms timed region (budget 20%) |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-direct | einsum_plan_read_into_accum_f64_s256 untimed 487.5 ms is 1510% of the 32.3 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_into_f64_s256 untimed 497.4 ms is 1010% of the 49.3 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_into_accum_f64_s256 untimed 480.0 ms is 1753% of the 27.4 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | tenferro-cuda-direct | sub_strided_read_f64_n1048576 untimed 180.7 ms is 1078% of the 16.8 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | tenferro-cuda-direct | div_strided_read_f64_n1048576 untimed 168.4 ms is 1003% of the 16.8 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | tenferro-cuda-direct | rem_strided_read_f64_n1048576 untimed 162.8 ms is 971% of the 16.8 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | tenferro-cuda-direct | pow_strided_read_f64_n1048576 untimed 170.8 ms is 976% of the 17.5 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-direct | maximum_strided_read_f64_n1048576 untimed 159.8 ms is 952% of the 16.8 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-direct | minimum_strided_read_f64_n1048576 untimed 159.3 ms is 949% of the 16.8 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | tenferro-cuda-direct | abs_strided_read_f64_n1048576 untimed 131.6 ms is 1442% of the 9.1 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | tenferro-cuda-direct | sign_strided_read_f64_n1048576 untimed 136.3 ms is 1489% of the 9.2 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-direct | expm1_strided_read_f64_n1048576 untimed 143.6 ms is 1551% of the 9.3 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | tenferro-cuda-direct | log_strided_read_f64_n1048576 untimed 147.5 ms is 1572% of the 9.4 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-direct | log1p_strided_read_f64_n1048576 untimed 164.1 ms is 1750% of the 9.4 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | tenferro-cuda-direct | erf_strided_read_f64_n1048576 untimed 174.7 ms is 1839% of the 9.5 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | tenferro-cuda-direct | sin_strided_read_f64_n1048576 untimed 160.5 ms is 1541% of the 10.4 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | tenferro-cuda-direct | cos_strided_read_f64_n1048576 untimed 158.1 ms is 1557% of the 10.2 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | sqrt_strided_read_f64_n1048576 untimed 157.4 ms is 1720% of the 9.2 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | rsqrt_strided_read_f64_n1048576 untimed 166.6 ms is 1822% of the 9.1 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_prod_strided_read_f64_n1048576 untimed 25.3 ms is 74% of the 34.5 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_max_strided_read_f64_n1048576 untimed 25.9 ms is 74% of the 35.2 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_min_strided_read_f64_n1048576 untimed 26.2 ms is 74% of the 35.2 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_sum_squares_strided_read_f64_n1048576 untimed 24.7 ms is 70% of the 35.1 ms timed region (budget 20%) |
| add_into_f64_n1048576 | tenferro-cuda-eager | add_into_f64_n1048576 untimed 714.9 ms is 11303% of the 6.3 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | tenferro-cuda-eager | add_read_into_f64_n1048576 untimed 699.9 ms is 11117% of the 6.3 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | tenferro-cuda-eager | sub_into_f64_n1048576 untimed 1107.7 ms is 17449% of the 6.3 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | tenferro-cuda-eager | sub_read_into_f64_n1048576 untimed 1079.0 ms is 17188% of the 6.3 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | tenferro-cuda-eager | mul_into_f64_n1048576 untimed 1145.5 ms is 18138% of the 6.3 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | tenferro-cuda-eager | mul_read_into_f64_n1048576 untimed 1293.8 ms is 20574% of the 6.3 ms timed region (budget 20%) |
| div_into_f64_n1048576 | tenferro-cuda-eager | div_into_f64_n1048576 untimed 864.9 ms is 13856% of the 6.2 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | tenferro-cuda-eager | div_read_into_f64_n1048576 untimed 709.8 ms is 11422% of the 6.2 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | tenferro-cuda-eager | neg_into_f64_n1048576 untimed 503.8 ms is 12194% of the 4.1 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | tenferro-cuda-eager | neg_read_into_f64_n1048576 untimed 478.3 ms is 11481% of the 4.2 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | tenferro-cuda-eager | conj_into_c64_n1048576 untimed 519.5 ms is 8651% of the 6.0 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | tenferro-cuda-eager | conj_read_into_c64_n1048576 untimed 528.9 ms is 8775% of the 6.0 ms timed region (budget 20%) |
| tensordot_f64_n262144 | tenferro-cuda-eager | tensordot_f64_n262144 untimed 2260.6 ms is 4177% of the 54.1 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | dot_general_batch_f64_s256 untimed 20.0 ms is 61% of the 32.8 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | dot_conj_l0r0_c64_n262144 untimed 17.3 ms is 35% of the 48.8 ms timed region (budget 20%) |
| add_into_f64_n1048576 | pytorch-cuda | add_into_f64_n1048576 untimed 1195.8 ms is 20419% of the 5.9 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | pytorch-cuda | add_read_into_f64_n1048576 untimed 710.3 ms is 12375% of the 5.7 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | pytorch-cuda | sub_into_f64_n1048576 untimed 238.9 ms is 4196% of the 5.7 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | pytorch-cuda | sub_read_into_f64_n1048576 untimed 243.7 ms is 4267% of the 5.7 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | pytorch-cuda | mul_into_f64_n1048576 untimed 272.6 ms is 4703% of the 5.8 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | pytorch-cuda | mul_read_into_f64_n1048576 untimed 265.7 ms is 4586% of the 5.8 ms timed region (budget 20%) |
| div_into_f64_n1048576 | pytorch-cuda | div_into_f64_n1048576 untimed 288.4 ms is 4538% of the 6.4 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | pytorch-cuda | div_read_into_f64_n1048576 untimed 291.7 ms is 4588% of the 6.4 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | pytorch-cuda | neg_into_f64_n1048576 untimed 541.1 ms is 14542% of the 3.7 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | pytorch-cuda | neg_read_into_f64_n1048576 untimed 513.3 ms is 13792% of the 3.7 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | pytorch-cuda | conj_into_c64_n1048576 untimed 619.3 ms is 10221% of the 6.1 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | pytorch-cuda | conj_read_into_c64_n1048576 untimed 606.6 ms is 10030% of the 6.0 ms timed region (budget 20%) |
| tensordot_f64_n262144 | pytorch-cuda | tensordot_f64_n262144 untimed 176.2 ms is 283% of the 62.3 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | pytorch-cuda | vdot_read_f64_n262144 untimed 20.9 ms is 52% of the 39.8 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | pytorch-cuda | norm_squared_read_f64_n262144 untimed 15.6 ms is 40% of the 38.7 ms timed region (budget 20%) |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | axpby_read_into_accum_f64_n1048576 untimed 489.8 ms is 6141% of the 8.0 ms timed region (budget 20%) |
| dot_general_read_into_f64_s256 | pytorch-cuda | dot_general_read_into_f64_s256 untimed 46.1 ms is 100% of the 46.0 ms timed region (budget 20%) |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | dot_general_read_into_accum_f64_s256 untimed 65.4 ms is 122% of the 53.4 ms timed region (budget 20%) |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | dot_general_read_into_accum_cached_f64_s256 untimed 47.2 ms is 91% of the 52.1 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | pytorch-cuda | add_strided_read_f64_n1048576 untimed 482.7 ms is 8187% of the 5.9 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | pytorch-cuda | mul_strided_read_f64_n1048576 untimed 238.0 ms is 4093% of the 5.8 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | pytorch-cuda | neg_strided_read_f64_n1048576 untimed 247.0 ms is 6551% of the 3.8 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | pytorch-cuda | exp_strided_read_f64_n1048576 untimed 268.3 ms is 6822% of the 3.9 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | tanh_strided_read_f64_n1048576 untimed 221.1 ms is 5643% of the 3.9 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_strided_read_f64_n1048576 untimed 45.4 ms is 88% of the 51.6 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | pytorch-cuda | conj_strided_read_c64_n1048576 untimed 324.7 ms is 5320% of the 6.1 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | pytorch-cuda | dot_general_batch_f64_s256 untimed 58.5 ms is 125% of the 46.7 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | dot_conj_l0r0_c64_n262144 untimed 18.1 ms is 36% of the 50.3 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | pytorch-cuda | einsum_plan_owned_f64_s256 untimed 21.3 ms is 43% of the 50.1 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | pytorch-cuda | einsum_plan_read_f64_s256 untimed 19.7 ms is 38% of the 51.4 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | einsum_plan_strided_read_f64_s256 untimed 20.5 ms is 40% of the 51.9 ms timed region (budget 20%) |
| einsum_plan_into_f64_s256 | pytorch-cuda | einsum_plan_into_f64_s256 untimed 18.6 ms is 45% of the 41.5 ms timed region (budget 20%) |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | einsum_plan_read_into_f64_s256 untimed 29.4 ms is 79% of the 37.5 ms timed region (budget 20%) |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_read_into_accum_f64_s256 untimed 46.5 ms is 91% of the 50.9 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_f64_s256 untimed 19.1 ms is 45% of the 42.3 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_accum_f64_s256 untimed 50.3 ms is 94% of the 53.2 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | pytorch-cuda | sub_strided_read_f64_n1048576 untimed 195.7 ms is 3358% of the 5.8 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | pytorch-cuda | div_strided_read_f64_n1048576 untimed 237.0 ms is 3686% of the 6.4 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | pytorch-cuda | rem_strided_read_f64_n1048576 untimed 251.0 ms is 4265% of the 5.9 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | pytorch-cuda | pow_strided_read_f64_n1048576 untimed 276.2 ms is 1967% of the 14.0 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | maximum_strided_read_f64_n1048576 untimed 287.8 ms is 4918% of the 5.9 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | minimum_strided_read_f64_n1048576 untimed 319.2 ms is 5469% of the 5.8 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | pytorch-cuda | abs_strided_read_f64_n1048576 untimed 639.4 ms is 16300% of the 3.9 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | pytorch-cuda | sign_strided_read_f64_n1048576 untimed 569.0 ms is 15315% of the 3.7 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | expm1_strided_read_f64_n1048576 untimed 591.8 ms is 12406% of the 4.8 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | pytorch-cuda | log_strided_read_f64_n1048576 untimed 608.5 ms is 10192% of the 6.0 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | log1p_strided_read_f64_n1048576 untimed 622.2 ms is 9967% of the 6.2 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | pytorch-cuda | erf_strided_read_f64_n1048576 untimed 464.4 ms is 6740% of the 6.9 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | pytorch-cuda | sin_strided_read_f64_n1048576 untimed 242.4 ms is 4617% of the 5.2 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | pytorch-cuda | cos_strided_read_f64_n1048576 untimed 304.8 ms is 5731% of the 5.3 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | sqrt_strided_read_f64_n1048576 untimed 256.1 ms is 6804% of the 3.8 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | rsqrt_strided_read_f64_n1048576 untimed 272.5 ms is 7294% of the 3.7 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | reduce_prod_strided_read_f64_n1048576 untimed 1172.5 ms is 2284% of the 51.3 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | reduce_max_strided_read_f64_n1048576 untimed 45.3 ms is 87% of the 51.9 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | reduce_min_strided_read_f64_n1048576 untimed 40.9 ms is 79% of the 51.8 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_squares_strided_read_f64_n1048576 untimed 36.9 ms is 91% of the 40.8 ms timed region (budget 20%) |
| tensordot_f64_n262144 | jax-cuda | tensordot_f64_n262144 untimed 3063.8 ms is 8861% of the 34.6 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | jax-cuda | vdot_read_f64_n262144 untimed 328.0 ms is 648% of the 50.6 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | jax-cuda | norm_squared_read_f64_n262144 untimed 20.4 ms is 37% of the 55.7 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | jax-cuda | add_strided_read_f64_n1048576 untimed 801.5 ms is 5708% of the 14.0 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | jax-cuda | mul_strided_read_f64_n1048576 untimed 674.5 ms is 2391% of the 28.2 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | jax-cuda | neg_strided_read_f64_n1048576 untimed 509.2 ms is 2182% of the 23.3 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | jax-cuda | exp_strided_read_f64_n1048576 untimed 328.5 ms is 1538% of the 21.4 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | jax-cuda | tanh_strided_read_f64_n1048576 untimed 341.6 ms is 1682% of the 20.3 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | reduce_sum_strided_read_f64_n1048576 untimed 194.7 ms is 439% of the 44.4 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | jax-cuda | conj_strided_read_c64_n1048576 untimed 635.8 ms is 6042% of the 10.5 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | jax-cuda | dot_general_batch_f64_s256 untimed 2643.1 ms is 6744% of the 39.2 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | jax-cuda | dot_conj_l0r0_c64_n262144 untimed 108.0 ms is 219% of the 49.4 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | jax-cuda | einsum_plan_owned_f64_s256 untimed 2734.1 ms is 6043% of the 45.2 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | jax-cuda | einsum_plan_read_f64_s256 untimed 27.4 ms is 61% of the 45.3 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | jax-cuda | einsum_plan_strided_read_f64_s256 untimed 360.9 ms is 699% of the 51.6 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | jax-cuda | sub_strided_read_f64_n1048576 untimed 783.8 ms is 3019% of the 26.0 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | jax-cuda | div_strided_read_f64_n1048576 untimed 527.9 ms is 2267% of the 23.3 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | jax-cuda | rem_strided_read_f64_n1048576 untimed 631.8 ms is 2844% of the 22.2 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | jax-cuda | pow_strided_read_f64_n1048576 untimed 592.8 ms is 1927% of the 30.8 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | jax-cuda | maximum_strided_read_f64_n1048576 untimed 681.4 ms is 2506% of the 27.2 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | jax-cuda | minimum_strided_read_f64_n1048576 untimed 624.5 ms is 2349% of the 26.6 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | jax-cuda | abs_strided_read_f64_n1048576 untimed 400.4 ms is 1881% of the 21.3 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | jax-cuda | sign_strided_read_f64_n1048576 untimed 381.9 ms is 1830% of the 20.9 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | jax-cuda | expm1_strided_read_f64_n1048576 untimed 430.5 ms is 1956% of the 22.0 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | jax-cuda | log_strided_read_f64_n1048576 untimed 370.4 ms is 1683% of the 22.0 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | jax-cuda | log1p_strided_read_f64_n1048576 untimed 348.7 ms is 1463% of the 23.8 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | jax-cuda | erf_strided_read_f64_n1048576 untimed 356.0 ms is 1151% of the 30.9 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | jax-cuda | sin_strided_read_f64_n1048576 untimed 402.6 ms is 1906% of the 21.1 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | jax-cuda | cos_strided_read_f64_n1048576 untimed 389.4 ms is 1797% of the 21.7 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | jax-cuda | sqrt_strided_read_f64_n1048576 untimed 334.6 ms is 1580% of the 21.2 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | rsqrt_strided_read_f64_n1048576 untimed 339.7 ms is 1046% of the 32.5 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | reduce_prod_strided_read_f64_n1048576 untimed 185.0 ms is 403% of the 45.9 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | reduce_max_strided_read_f64_n1048576 untimed 253.3 ms is 540% of the 46.9 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | reduce_min_strided_read_f64_n1048576 untimed 123.6 ms is 227% of the 54.4 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | reduce_sum_squares_strided_read_f64_n1048576 untimed 252.3 ms is 451% of the 56.0 ms timed region (budget 20%) |