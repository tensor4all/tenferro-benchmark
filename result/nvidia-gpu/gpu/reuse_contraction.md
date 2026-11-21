# gpu/reuse_contraction: CUDA public API benchmark

Measured cells: **150**. Unsupported cells: **74**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1415.5 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261121_082311/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**22** of 56) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Transposed borrowed-read cases use prepared native CUDA views in tenferro and PyTorch. JAX compares the same logical values in its native representation; its columns do not claim an equivalent public strided-view contract. View creation and input conversion are outside timing.

Generalized batch dot preserves native output order: tenferro [m,n,b], PyTorch/JAX [b,m,n]. Input permutation views and dimension-number configurations are prepared before timing. Output axes are normalized only on downloaded CPU arrays for verification; no timed output transpose or copy is added.

Prepared einsum uses the public ConcreteEinsumPlan for ik,kj->ij, built once outside execution and sampling. Native matrix multiplication/reuse/accumulation references match the selected mathematical and output-ownership contracts; they do not assert an equivalent prepared-plan interface. Transposed input views/values, read/write handles and accumulation seed restoration are untimed.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| add_into_f64_n1048576 | 21.730 (0.166) | unsupported | 12.121 (0.183) | unsupported |
| add_read_into_f64_n1048576 | 21.829 (0.429) | unsupported | 12.241 (0.255) | unsupported |
| sub_into_f64_n1048576 | 21.793 (0.611) | unsupported | 12.121 (0.443) | unsupported |
| sub_read_into_f64_n1048576 | 21.765 (0.575) | unsupported | 12.279 (0.055) | unsupported |
| mul_into_f64_n1048576 | 21.618 (0.343) | unsupported | 11.998 (0.246) | unsupported |
| mul_read_into_f64_n1048576 | 21.739 (0.203) | unsupported | 12.518 (0.203) | unsupported |
| div_into_f64_n1048576 | 21.753 (0.278) | unsupported | 12.995 (0.168) | unsupported |
| div_read_into_f64_n1048576 | 22.561 (0.311) | unsupported | 13.076 (0.161) | unsupported |
| neg_into_f64_n1048576 | 18.954 (0.331) | unsupported | 7.787 (0.037) | unsupported |
| neg_read_into_f64_n1048576 | 18.989 (0.178) | unsupported | 8.078 (0.561) | unsupported |
| conj_into_c64_n1048576 | 33.330 (0.740) | unsupported | 25.597 (0.507) | unsupported |
| conj_read_into_c64_n1048576 | 33.687 (0.204) | unsupported | 25.009 (0.262) | unsupported |
| tensordot_f64_n262144 † | 44.134 (0.814) | 54.086 (2.879) | 25.209 (0.549) | 156.141 (65.069) |
| vdot_read_f64_n262144 † | 64.183 (0.567) | unsupported | 9.563 (0.098) | 75.940 (1.412) |
| norm_squared_read_f64_n262144 † | 24.030 (0.204) | unsupported | 9.168 (0.012) | 69.646 (5.459) |
| axpby_read_into_accum_f64_n1048576 | 33.071 (2.120) | unsupported | 16.600 (0.160) | unsupported |
| dot_general_read_into_f64_s256 † | 17.329 (0.123) | unsupported | 8.301 (0.094) | unsupported |
| dot_general_read_into_accum_f64_s256 † | 19.858 (0.146) | unsupported | 9.298 (0.165) | unsupported |
| dot_general_read_into_accum_cached_f64_s256 † | 19.875 (0.157) | unsupported | 9.256 (0.154) | unsupported |
| add_strided_read_f64_n1048576 † | 73.073 (0.247) | unsupported | 12.318 (0.190) | 83.391 (3.755) |
| mul_strided_read_f64_n1048576 | 73.363 (9.098) | unsupported | 12.276 (0.131) | 81.184 (2.092) |
| neg_strided_read_f64_n1048576 | 44.460 (0.455) | unsupported | 7.833 (0.033) | 79.424 (3.239) |
| exp_strided_read_f64_n1048576 | 42.521 (0.344) | unsupported | 8.315 (0.067) | 57.451 (4.086) |
| tanh_strided_read_f64_n1048576 | 42.555 (0.263) | unsupported | 8.193 (0.025) | 50.103 (4.256) |
| reduce_sum_strided_read_f64_n1048576 † | 64.217 (0.450) | unsupported | 13.451 (0.022) | 79.636 (2.165) |
| conj_strided_read_c64_n1048576 | 63.796 (1.449) | unsupported | 25.207 (0.236) | 46.023 (4.325) |
| dot_general_batch_f64_s256 † | 26.366 (0.349) | 29.209 (0.689) | 18.059 (0.277) | 79.657 (1.396) |
| dot_conj_l0r0_c64_n262144 † | 37.039 (0.316) | 36.970 (0.198) | 12.499 (0.225) | 52.856 (3.176) |
| einsum_plan_owned_f64_s256 † | 23.978 (0.242) | unsupported | 9.355 (0.179) | 49.408 (3.202) |
| einsum_plan_read_f64_s256 † | 23.892 (0.243) | unsupported | 9.259 (0.105) | 48.759 (1.207) |
| einsum_plan_strided_read_f64_s256 † | 63.803 (0.677) | unsupported | 9.287 (0.234) | 48.335 (1.546) |
| einsum_plan_into_f64_s256 † | 16.366 (0.318) | unsupported | 8.075 (0.008) | unsupported |
| einsum_plan_read_into_f64_s256 † | 17.391 (0.147) | unsupported | 8.070 (0.007) | unsupported |
| einsum_plan_read_into_accum_f64_s256 † | 18.798 (0.109) | unsupported | 9.198 (0.113) | unsupported |
| einsum_plan_strided_read_into_f64_s256 † | 57.301 (0.490) | unsupported | 7.964 (0.037) | unsupported |
| einsum_plan_strided_read_into_accum_f64_s256 † | 59.180 (1.700) | unsupported | 9.464 (0.194) | unsupported |
| sub_strided_read_f64_n1048576 | 72.533 (0.496) | unsupported | 12.440 (0.248) | 57.192 (2.526) |
| div_strided_read_f64_n1048576 | 72.667 (0.774) | unsupported | 12.899 (0.173) | 49.579 (0.888) |
| rem_strided_read_f64_n1048576 | 94.986 (0.860) | unsupported | 12.469 (0.212) | 47.348 (0.843) |
| pow_strided_read_f64_n1048576 | 95.955 (0.393) | unsupported | 29.266 (0.330) | 59.046 (1.428) |
| maximum_strided_read_f64_n1048576 | 82.564 (3.882) | unsupported | 12.454 (0.163) | 59.863 (1.392) |
| minimum_strided_read_f64_n1048576 | 82.468 (1.125) | unsupported | 12.245 (0.149) | 64.243 (2.606) |
| abs_strided_read_f64_n1048576 | 42.919 (0.879) | unsupported | 10.437 (3.785) | 55.347 (3.301) |
| sign_strided_read_f64_n1048576 | 42.468 (0.199) | unsupported | 7.847 (0.068) | 60.032 (3.499) |
| expm1_strided_read_f64_n1048576 | 43.961 (0.414) | unsupported | 9.834 (0.070) | 74.466 (1.518) |
| log_strided_read_f64_n1048576 | 46.537 (0.336) | unsupported | 12.619 (0.056) | 60.264 (1.745) |
| log1p_strided_read_f64_n1048576 | 46.458 (0.189) | unsupported | 13.114 (0.245) | 73.307 (2.184) |
| erf_strided_read_f64_n1048576 | 48.792 (0.434) | unsupported | 14.344 (0.042) | 90.878 (1.855) |
| sin_strided_read_f64_n1048576 | 44.457 (0.127) | unsupported | 11.192 (0.035) | 59.581 (11.252) |
| cos_strided_read_f64_n1048576 | 46.463 (0.299) | unsupported | 11.316 (0.045) | 59.241 (1.339) |
| sqrt_strided_read_f64_n1048576 | 42.330 (0.403) | unsupported | 7.942 (0.037) | 60.071 (2.073) |
| rsqrt_strided_read_f64_n1048576 | 44.255 (0.406) | unsupported | 7.958 (0.049) | 91.966 (2.659) |
| reduce_prod_strided_read_f64_n1048576 † | 63.721 (0.359) | unsupported | 13.424 (0.023) | 69.758 (1.487) |
| reduce_max_strided_read_f64_n1048576 † | 64.045 (0.287) | unsupported | 13.505 (0.014) | 134.228 (3.241) |
| reduce_min_strided_read_f64_n1048576 † | 64.223 (0.491) | unsupported | 13.531 (0.015) | 130.945 (37.078) |
| reduce_sum_squares_strided_read_f64_n1048576 † | 64.035 (0.130) | unsupported | 21.294 (0.020) | 1351.743 (166.802) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.771 | 16777216 | 0.214 | 0 |
| abs_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.334 | 16777216 | 1.136 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.373 | 16777216 | 0.276 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| add_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.388 | 25165824 | 1.467 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.695 | 25165824 | 0.818 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.392 | 25165824 | 1.452 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.699 | 25165824 | 0.814 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 1.334 | 25165824 | 0.213 | 0 |
| add_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.394 | 25165824 | 1.443 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.338 | 25165824 | 0.243 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| axpby_read_into_accum_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.531 | 25165824 | 1.071 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.058 | 25165824 | 0.538 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No matching public caller-owned accumulating output API; donation is not output reuse. |
| conj_into_c64_n1048576 | jax-cuda | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.410 | 33554432 | 0.926 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.533 | 33554432 | 0.711 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1048576 | jax-cuda | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.400 | 33554432 | 0.948 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.539 | 33554432 | 0.704 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_strided_read_c64_n1048576 | jax-cuda | 16 | 15 | 0.736 | 33554432 | 0.515 | 0 |
| conj_strided_read_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.403 | 33554432 | 0.940 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.021 | 33554432 | 0.372 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| cos_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.896 | 16777216 | 0.200 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.362 | 16777216 | 1.047 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.487 | 16777216 | 0.255 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| div_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.416 | 25165824 | 1.368 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.696 | 25165824 | 0.817 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.418 | 25165824 | 1.360 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.722 | 25165824 | 0.788 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.587 | 25165824 | 0.359 | 0 |
| div_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.413 | 25165824 | 1.378 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.325 | 25165824 | 0.245 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| dot_conj_l0r0_c64_n262144 | jax-cuda | 64 | 15 | 3.383 | 8388624 | 0.112 | 1.221e-05 |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | 256 | 15 | 3.200 | 8388624 | 0.474 | 1.366e-05 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | 64 | 15 | 2.371 | 8388624 | 0.160 | 1.212e-05 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | 64 | 15 | 2.366 | 8388624 | 0.160 | 1.212e-05 |
| dot_general_batch_f64_s256 | jax-cuda | 32 | 15 | 2.549 | 3210240 | 0.028 | 7.463e-07 |
| dot_general_batch_f64_s256 | pytorch-cuda | 128 | 15 | 2.312 | 3210240 | 0.126 | 7.463e-07 |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | 128 | 15 | 3.375 | 3210240 | 0.086 | 7.463e-07 |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | 64 | 15 | 1.869 | 3210240 | 0.078 | 7.463e-07 |
| dot_general_read_into_accum_cached_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | 256 | 15 | 2.369 | 2097152 | 0.160 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.544 | 2097152 | 0.075 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.380 | 2097152 | 0.159 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.542 | 2097152 | 0.075 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.125 | 2097152 | 0.178 | 0 |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.218 | 2097152 | 0.085 | 0 |
| dot_general_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| einsum_plan_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.067 | 1070080 | 0.094 | 0 |
| einsum_plan_into_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.095 | 1070080 | 0.046 | 0 |
| einsum_plan_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_owned_f64_s256 | jax-cuda | 64 | 15 | 3.162 | 1070080 | 0.015 | 0 |
| einsum_plan_owned_f64_s256 | pytorch-cuda | 256 | 15 | 2.395 | 1070080 | 0.081 | 0 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-direct | 128 | 15 | 3.069 | 1070080 | 0.032 | 0 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_f64_s256 | jax-cuda | 64 | 15 | 3.121 | 1070080 | 0.016 | 0 |
| einsum_plan_read_f64_s256 | pytorch-cuda | 256 | 15 | 2.370 | 1070080 | 0.082 | 0 |
| einsum_plan_read_f64_s256 | tenferro-cuda-direct | 128 | 15 | 3.058 | 1070080 | 0.032 | 0 |
| einsum_plan_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_accum_f64_s256 | jax-cuda | — | — | — | 1608704 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.355 | 1608704 | 0.124 | 0 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.406 | 1608704 | 0.060 | 0 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.066 | 1070080 | 0.094 | 0 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.226 | 1070080 | 0.043 | 0 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_f64_s256 | jax-cuda | 64 | 15 | 3.093 | 1070080 | 0.016 | 0 |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | 256 | 15 | 2.377 | 1070080 | 0.081 | 0 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-direct | 32 | 15 | 2.042 | 1070080 | 0.012 | 0 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_accum_f64_s256 | jax-cuda | — | — | — | 1608704 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.423 | 1608704 | 0.120 | 0 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-direct | 32 | 15 | 1.894 | 1608704 | 0.019 | 0 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.039 | 1070080 | 0.095 | 0 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-direct | 64 | 15 | 3.667 | 1070080 | 0.013 | 0 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| erf_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.908 | 16777216 | 0.130 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.459 | 16777216 | 0.826 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.561 | 16777216 | 0.243 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| exp_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.838 | 16777216 | 0.206 | 1.095e-07 |
| exp_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.266 | 16777216 | 1.425 | 1.095e-07 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.361 | 16777216 | 0.279 | 1.095e-07 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| expm1_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.383 | 16777216 | 0.159 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.315 | 16777216 | 1.205 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.407 | 16777216 | 0.270 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log1p_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.346 | 16777216 | 0.162 | 1.11e-07 |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.420 | 16777216 | 0.904 | 1.11e-07 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.487 | 16777216 | 0.255 | 1.11e-07 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.928 | 16777216 | 0.197 | 4.308e-08 |
| log_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.404 | 16777216 | 0.939 | 4.308e-08 |
| log_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.489 | 16777216 | 0.255 | 4.308e-08 |
| log_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| maximum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.916 | 25165824 | 0.297 | 0 |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.399 | 25165824 | 1.428 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.642 | 25165824 | 0.215 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| minimum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.056 | 25165824 | 0.277 | 0 |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.392 | 25165824 | 1.452 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.639 | 25165824 | 0.216 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| mul_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.384 | 25165824 | 1.482 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.692 | 25165824 | 0.822 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.401 | 25165824 | 1.420 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.696 | 25165824 | 0.818 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.598 | 25165824 | 0.219 | 0 |
| mul_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.393 | 25165824 | 1.448 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.348 | 25165824 | 0.242 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| neg_into_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.249 | 16777216 | 1.522 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.607 | 16777216 | 0.625 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.258 | 16777216 | 1.467 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.608 | 16777216 | 0.624 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.542 | 16777216 | 0.149 | 0 |
| neg_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.251 | 16777216 | 1.513 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.423 | 16777216 | 0.267 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| norm_squared_read_f64_n262144 | jax-cuda | 32 | 15 | 2.229 | 2097160 | 0.021 | 1.664e-07 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.347 | 2097160 | 0.162 | 1.664e-07 |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | 128 | 15 | 3.076 | 2097160 | 0.062 | 1.664e-07 |
| norm_squared_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 2097160 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| pow_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.889 | 25165824 | 0.301 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.937 | 25165824 | 0.607 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.071 | 25165824 | 0.185 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.148 | 8392704 | 0.044 | 0 |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.457 | 8392704 | 0.439 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.049 | 8392704 | 0.093 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.095 | 8392704 | 0.045 | 0 |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.464 | 8392704 | 0.438 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.055 | 8392704 | 0.092 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.232 | 8392704 | 0.085 | 5.439e-06 |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.436 | 8392704 | 0.442 | 4.106e-06 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.039 | 8392704 | 0.093 | 4.661e-06 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | 2 | 15 | 2.703 | 8392704 | 0.004 | 3.311e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.726 | 8392704 | 0.278 | 3.309e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.049 | 8392704 | 0.093 | 3.312e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.548 | 8392704 | 0.074 | 7.159e-06 |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.444 | 8392704 | 0.441 | 5.935e-06 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.055 | 8392704 | 0.092 | 4.724e-06 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rem_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.515 | 25165824 | 0.375 | 0 |
| rem_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.399 | 25165824 | 1.426 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.040 | 25165824 | 0.187 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.943 | 16777216 | 0.129 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.255 | 16777216 | 1.489 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.416 | 16777216 | 0.268 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sign_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.921 | 16777216 | 0.197 | 0 |
| sign_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.251 | 16777216 | 1.511 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.359 | 16777216 | 0.279 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sin_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.907 | 16777216 | 0.199 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.358 | 16777216 | 1.059 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.423 | 16777216 | 0.267 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sqrt_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.922 | 16777216 | 0.197 | 0 |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.254 | 16777216 | 1.492 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.355 | 16777216 | 0.280 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sub_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.388 | 25165824 | 1.467 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.697 | 25165824 | 0.816 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.393 | 25165824 | 1.448 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.696 | 25165824 | 0.817 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.830 | 25165824 | 0.311 | 0 |
| sub_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.398 | 25165824 | 1.429 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.321 | 25165824 | 0.245 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tanh_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.603 | 16777216 | 0.237 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.262 | 16777216 | 1.447 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.362 | 16777216 | 0.279 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tensordot_f64_n262144 | jax-cuda | 16 | 15 | 2.498 | 6291456 | 0.028 | 6.714e-06 |
| tensordot_f64_n262144 | pytorch-cuda | 128 | 15 | 3.227 | 6291456 | 0.176 | 6.714e-06 |
| tensordot_f64_n262144 | tenferro-cuda-direct | 64 | 15 | 2.825 | 6291456 | 0.101 | 6.714e-06 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 8 | 15 | 0.433 | 6291456 | 0.082 | 6.714e-06 |
| vdot_read_f64_n262144 | jax-cuda | 32 | 15 | 2.430 | 4194312 | 0.039 | 2.995e-06 |
| vdot_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.448 | 4194312 | 0.310 | 2.995e-06 |
| vdot_read_f64_n262144 | tenferro-cuda-direct | 32 | 15 | 2.054 | 4194312 | 0.046 | 2.995e-06 |
| vdot_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 4194312 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261121_082311/gpu_reuse_contraction.yaml
timestamp: '2026-11-21T08:23:11+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: ce9125d0cff4ee40841120759b5df45252c27f22
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: a741cd656e4d51e77d8369154bf789e5f0147f5c
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
  cpu: AMD EPYC 7543 32-Core Processor
  logical_cpus: 128
  power:
    cpufreq_boost: '1'
    governor: schedutil
    scaling_max_freq_khz: '2800000'
    smt:
      active: '1'
      control: 'on'
  numa:
    nodes: 2
    cpulists:
      '0': 0-31,64-95
      '1': 32-63,96-127
  smt_siblings:
  - 0,64
  - 1,65
  - 2,66
  - 3,67
  - 4,68
  - 5,69
  - 6,70
  - 7,71
  - 8,72
  - 9,73
  - 10,74
  - 11,75
  - 12,76
  - 13,77
  - 14,78
  - 15,79
  - 16,80
  - 17,81
  - 18,82
  - 19,83
  - 20,84
  - 21,85
  - 22,86
  - 23,87
  - 24,88
  - 25,89
  - 26,90
  - 27,91
  - 28,92
  - 29,93
  - 30,94
  - 31,95
  - 32,96
  - 33,97
  - 34,98
  - 35,99
  - 36,100
  - 37,101
  - 38,102
  - 39,103
  - 40,104
  - 41,105
  - 42,106
  - 43,107
  - 44,108
  - 45,109
  - 46,110
  - 47,111
  - 48,112
  - 49,113
  - 50,114
  - 51,115
  - 52,116
  - 53,117
  - 54,118
  - 55,119
  - 56,120
  - 57,121
  - 58,122
  - 59,123
  - 60,124
  - 61,125
  - 62,126
  - 63,127
  l3_domains:
  - 0-3,64-67
  - 4-7,68-71
  - 8-11,72-75
  - 12-15,76-79
  - 16-19,80-83
  - 20-23,84-87
  - 24-27,88-91
  - 28-31,92-95
  - 32-35,96-99
  - 36-39,100-103
  - 40-43,104-107
  - 44-47,108-111
  - 48-51,112-115
  - 52-55,116-119
  - 56-59,120-123
  - 60-63,124-127
build:
  toolchain:
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: 7894e3c40841
  os: Linux-6.8.0-138-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-127
    l3_domains:
    - 0-3,64-67
    - 4-7,68-71
    - 8-11,72-75
    - 12-15,76-79
    - 16-19,80-83
    - 20-23,84-87
    - 24-27,88-91
    - 28-31,92-95
    - 32-35,96-99
    - 36-39,100-103
    - 40-43,104-107
    - 44-47,108-111
    - 48-51,112-115
    - 52-55,116-119
    - 56-59,120-123
    - 60-63,124-127
    cpu_boost: enabled
    scaling_governor: schedutil
    scaling_driver: acpi-cpufreq
  env:
    OMP_NUM_THREADS: '1'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    OPENBLAS_NUM_THREADS: '1'
    MATMUL_NUM_THREADS: '1'
    CUDA_HOME: /opt/tenferro-benchmark-ci/cuda
    CUDA_PATH: /opt/tenferro-benchmark-ci/cuda
    TENFERRO_CUTENSOR_PATH: /opt/tenferro-benchmark-ci/cutensor/lib/libcutensor.so.2
    GPU_RUNTIME_SHA256: 3d603f753ad72a63b928ce8e1b63847b8a21f94f0919276124993a6548126185
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
  device_uuid: GPU-98d8618e-77c8-160a-92f2-117e34fd43df
  device_memory_bytes: 85899345920
  driver_version: 595.112.07
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
  - 0.093871
  - 0.092601
  - 0.095741
  - 0.094821
  - 0.091741
  - 0.095801
  - 0.09431
  - 0.093421
  - 0.095691
  - 0.094681
  - 0.094361
  - 0.095031
  - 0.094901
  - 0.097871
  - 0.09506
  median_ms: 0.094821
  bytes_per_second: 1415485261703.6309
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-21T08:35:52.949737+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-98d8618e-77c8-160a-92f2-117e34fd43df
  cuda_visible_devices: GPU-98d8618e-77c8-160a-92f2-117e34fd43df
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
    src/bin/benchmark_gpu_public_api.rs: a5ed803c89dc3085b18ed3220a6d484facf179c9b55fe13042beb19d83090196
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 3d5bedcc3620f75cab2687fdf49210df324021d043b23aec6e169642ceff15ba
    src/bin/benchmark_gpu_public_api/multi.rs: 950cb1b3c1a7cca37665b58e3de074bd906dafb8c84b17a51c27ce797a73afa8
    src/bin/benchmark_gpu_public_api/plan.rs: 016e752c59edefc3ded1405c36325e8680736860cb65954919ea4cf0f3eed061
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: e9b588e54fd067e570b557a4b06e40f0b9b432d665f50b64ee7f2ff7f461754b
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: 88038508963fbc13303fb54b3898808e4303131321cabedb9f38255941ead438
    src/bin/benchmark_gpu_public_api/state.rs: e1cc2781ce7392154f32cc19372c9fc41ae29765aedc544c7ae7bf02f873c12b
    src/bin/benchmark_gpu_public_api/structural.rs: 10850458a640d11e79262c215072482ee463bc4ca0604f40d82ae56f345f6564
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: d1ff68f86496d991bcbc730138fe6e3076001b3f23cc1a94c44b96f6857cb8ed
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 7995bcf409258185e07210969a738896f64860b67df251fcd1100d11128a5431
    data/results/nvidia-gpu/gpu/reuse_contraction/20261121_082311/fixtures/manifest.json: 63a15555590136dc9184229fe03dd2eea2bd9e21f80607686d700c24c41fde1f
    target/release/benchmark_gpu_public_api: c67dd59529f9c6644708d8bc0a5a5fabc82f29fff70f5532c20f6a3066720813
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:36:32Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:36:21Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:36:10Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:36:13Z'
  compile:
    case_wall_seconds: 30.491
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
| add_into_f64_n1048576 | tenferro-cuda-direct | add_into_f64_n1048576 untimed 441.0 ms is 4226% of the 10.4 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | add_read_into_f64_n1048576 untimed 356.8 ms is 2439% of the 14.6 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | tenferro-cuda-direct | sub_into_f64_n1048576 untimed 364.7 ms is 3381% of the 10.8 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | sub_read_into_f64_n1048576 untimed 345.8 ms is 3223% of the 10.7 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | tenferro-cuda-direct | mul_into_f64_n1048576 untimed 342.4 ms is 3282% of the 10.4 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | mul_read_into_f64_n1048576 untimed 320.3 ms is 3064% of the 10.5 ms timed region (budget 20%) |
| div_into_f64_n1048576 | tenferro-cuda-direct | div_into_f64_n1048576 untimed 320.1 ms is 3042% of the 10.5 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | div_read_into_f64_n1048576 untimed 343.0 ms is 3164% of the 10.8 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | tenferro-cuda-direct | neg_into_f64_n1048576 untimed 347.1 ms is 3819% of the 9.1 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | neg_read_into_f64_n1048576 untimed 339.2 ms is 3713% of the 9.1 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | tenferro-cuda-direct | conj_into_c64_n1048576 untimed 344.6 ms is 4176% of the 8.3 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | conj_read_into_c64_n1048576 untimed 327.6 ms is 4033% of the 8.1 ms timed region (budget 20%) |
| tensordot_f64_n262144 | tenferro-cuda-direct | tensordot_f64_n262144 untimed 2031.8 ms is 4759% of the 42.7 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | tenferro-cuda-direct | vdot_read_f64_n262144 untimed 11.5 ms is 37% of the 30.8 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | norm_squared_read_f64_n262144 untimed 10.0 ms is 21% of the 46.6 ms timed region (budget 20%) |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | axpby_read_into_accum_f64_n1048576 untimed 357.2 ms is 2243% of the 15.9 ms timed region (budget 20%) |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | dot_general_read_into_f64_s256 untimed 279.2 ms is 836% of the 33.4 ms timed region (budget 20%) |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | dot_general_read_into_accum_f64_s256 untimed 308.2 ms is 809% of the 38.1 ms timed region (budget 20%) |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | dot_general_read_into_accum_cached_f64_s256 untimed 307.3 ms is 801% of the 38.4 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | add_strided_read_f64_n1048576 untimed 124.9 ms is 356% of the 35.1 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | mul_strided_read_f64_n1048576 untimed 115.1 ms is 305% of the 37.8 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | neg_strided_read_f64_n1048576 untimed 101.6 ms is 476% of the 21.3 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | exp_strided_read_f64_n1048576 untimed 90.2 ms is 441% of the 20.5 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | tanh_strided_read_f64_n1048576 untimed 93.0 ms is 455% of the 20.5 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_sum_strided_read_f64_n1048576 untimed 20.7 ms is 67% of the 30.8 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | conj_strided_read_c64_n1048576 untimed 141.2 ms is 904% of the 15.6 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | dot_general_batch_f64_s256 untimed 33.2 ms is 65% of the 50.8 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | dot_conj_l0r0_c64_n262144 untimed 19.9 ms is 56% of the 35.6 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | tenferro-cuda-direct | einsum_plan_owned_f64_s256 untimed 9.8 ms is 21% of the 46.4 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | tenferro-cuda-direct | einsum_plan_read_f64_s256 untimed 9.3 ms is 20% of the 46.0 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_f64_s256 untimed 6.5 ms is 21% of the 31.0 ms timed region (budget 20%) |
| einsum_plan_into_f64_s256 | tenferro-cuda-direct | einsum_plan_into_f64_s256 untimed 429.1 ms is 1346% of the 31.9 ms timed region (budget 20%) |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-direct | einsum_plan_read_into_f64_s256 untimed 447.1 ms is 1338% of the 33.4 ms timed region (budget 20%) |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-direct | einsum_plan_read_into_accum_f64_s256 untimed 467.9 ms is 1292% of the 36.2 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_into_f64_s256 untimed 451.6 ms is 814% of the 55.5 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_into_accum_f64_s256 untimed 451.8 ms is 1566% of the 28.8 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | tenferro-cuda-direct | sub_strided_read_f64_n1048576 untimed 119.9 ms is 344% of the 34.9 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | tenferro-cuda-direct | div_strided_read_f64_n1048576 untimed 126.3 ms is 362% of the 34.9 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | tenferro-cuda-direct | rem_strided_read_f64_n1048576 untimed 125.1 ms is 274% of the 45.6 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | tenferro-cuda-direct | pow_strided_read_f64_n1048576 untimed 135.2 ms is 294% of the 46.1 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-direct | maximum_strided_read_f64_n1048576 untimed 130.3 ms is 325% of the 40.1 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-direct | minimum_strided_read_f64_n1048576 untimed 121.9 ms is 307% of the 39.7 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | tenferro-cuda-direct | abs_strided_read_f64_n1048576 untimed 99.6 ms is 485% of the 20.6 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | tenferro-cuda-direct | sign_strided_read_f64_n1048576 untimed 93.6 ms is 459% of the 20.4 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-direct | expm1_strided_read_f64_n1048576 untimed 92.5 ms is 438% of the 21.1 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | tenferro-cuda-direct | log_strided_read_f64_n1048576 untimed 89.3 ms is 399% of the 22.4 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-direct | log1p_strided_read_f64_n1048576 untimed 102.8 ms is 461% of the 22.3 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | tenferro-cuda-direct | erf_strided_read_f64_n1048576 untimed 117.0 ms is 500% of the 23.4 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | tenferro-cuda-direct | sin_strided_read_f64_n1048576 untimed 91.0 ms is 426% of the 21.4 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | tenferro-cuda-direct | cos_strided_read_f64_n1048576 untimed 94.5 ms is 423% of the 22.3 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | sqrt_strided_read_f64_n1048576 untimed 90.2 ms is 444% of the 20.3 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | rsqrt_strided_read_f64_n1048576 untimed 106.4 ms is 501% of the 21.3 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_prod_strided_read_f64_n1048576 untimed 19.0 ms is 62% of the 30.6 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_max_strided_read_f64_n1048576 untimed 17.8 ms is 58% of the 30.8 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_min_strided_read_f64_n1048576 untimed 17.9 ms is 58% of the 30.9 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_sum_squares_strided_read_f64_n1048576 untimed 18.1 ms is 59% of the 30.8 ms timed region (budget 20%) |
| tensordot_f64_n262144 | tenferro-cuda-eager | tensordot_f64_n262144 untimed 2067.8 ms is 20619% of the 10.0 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | dot_general_batch_f64_s256 untimed 15.8 ms is 55% of the 28.7 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | dot_conj_l0r0_c64_n262144 untimed 18.0 ms is 51% of the 35.5 ms timed region (budget 20%) |
| add_into_f64_n1048576 | pytorch-cuda | add_into_f64_n1048576 untimed 202.4 ms is 3485% of the 5.8 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | pytorch-cuda | add_read_into_f64_n1048576 untimed 74.1 ms is 1235% of the 6.0 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | pytorch-cuda | sub_into_f64_n1048576 untimed 62.0 ms is 1081% of the 5.7 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | pytorch-cuda | sub_read_into_f64_n1048576 untimed 62.0 ms is 1052% of the 5.9 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | pytorch-cuda | mul_into_f64_n1048576 untimed 67.9 ms is 1175% of the 5.8 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | pytorch-cuda | mul_read_into_f64_n1048576 untimed 59.6 ms is 970% of the 6.1 ms timed region (budget 20%) |
| div_into_f64_n1048576 | pytorch-cuda | div_into_f64_n1048576 untimed 68.6 ms is 1097% of the 6.3 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | pytorch-cuda | div_read_into_f64_n1048576 untimed 59.9 ms is 952% of the 6.3 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | pytorch-cuda | neg_into_f64_n1048576 untimed 74.2 ms is 1984% of the 3.7 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | pytorch-cuda | neg_read_into_f64_n1048576 untimed 71.4 ms is 1826% of the 3.9 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | pytorch-cuda | conj_into_c64_n1048576 untimed 109.5 ms is 1784% of the 6.1 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | pytorch-cuda | conj_read_into_c64_n1048576 untimed 94.3 ms is 1565% of the 6.0 ms timed region (budget 20%) |
| tensordot_f64_n262144 | pytorch-cuda | tensordot_f64_n262144 untimed 64.0 ms is 131% of the 48.8 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | pytorch-cuda | vdot_read_f64_n262144 untimed 11.4 ms is 31% of the 36.7 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | pytorch-cuda | norm_squared_read_f64_n262144 untimed 7.5 ms is 21% of the 35.2 ms timed region (budget 20%) |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | axpby_read_into_accum_f64_n1048576 untimed 71.0 ms is 892% of the 8.0 ms timed region (budget 20%) |
| dot_general_read_into_f64_s256 | pytorch-cuda | dot_general_read_into_f64_s256 untimed 35.2 ms is 110% of the 32.0 ms timed region (budget 20%) |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | dot_general_read_into_accum_f64_s256 untimed 56.9 ms is 124% of the 45.9 ms timed region (budget 20%) |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | dot_general_read_into_accum_cached_f64_s256 untimed 35.3 ms is 99% of the 35.8 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | pytorch-cuda | add_strided_read_f64_n1048576 untimed 89.2 ms is 1503% of the 5.9 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | pytorch-cuda | mul_strided_read_f64_n1048576 untimed 86.3 ms is 1467% of the 5.9 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | pytorch-cuda | neg_strided_read_f64_n1048576 untimed 69.4 ms is 1844% of the 3.8 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | pytorch-cuda | exp_strided_read_f64_n1048576 untimed 90.8 ms is 2265% of the 4.0 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | tanh_strided_read_f64_n1048576 untimed 69.6 ms is 1768% of the 3.9 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_strided_read_f64_n1048576 untimed 26.2 ms is 50% of the 52.1 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | pytorch-cuda | conj_strided_read_c64_n1048576 untimed 139.3 ms is 2296% of the 6.1 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | pytorch-cuda | dot_general_batch_f64_s256 untimed 32.4 ms is 94% of the 34.6 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | dot_conj_l0r0_c64_n262144 untimed 11.3 ms is 23% of the 48.9 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | pytorch-cuda | einsum_plan_owned_f64_s256 untimed 11.9 ms is 33% of the 35.9 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | pytorch-cuda | einsum_plan_read_f64_s256 untimed 10.9 ms is 31% of the 35.6 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | einsum_plan_strided_read_f64_s256 untimed 11.1 ms is 31% of the 35.9 ms timed region (budget 20%) |
| einsum_plan_into_f64_s256 | pytorch-cuda | einsum_plan_into_f64_s256 untimed 9.9 ms is 32% of the 31.0 ms timed region (budget 20%) |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | einsum_plan_read_into_f64_s256 untimed 24.0 ms is 77% of the 31.0 ms timed region (budget 20%) |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_read_into_accum_f64_s256 untimed 32.9 ms is 93% of the 35.4 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_f64_s256 untimed 10.0 ms is 33% of the 30.6 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_accum_f64_s256 untimed 38.3 ms is 106% of the 36.2 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | pytorch-cuda | sub_strided_read_f64_n1048576 untimed 79.0 ms is 1298% of the 6.1 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | pytorch-cuda | div_strided_read_f64_n1048576 untimed 86.8 ms is 1397% of the 6.2 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | pytorch-cuda | rem_strided_read_f64_n1048576 untimed 103.9 ms is 1736% of the 6.0 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | pytorch-cuda | pow_strided_read_f64_n1048576 untimed 108.6 ms is 770% of the 14.1 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | maximum_strided_read_f64_n1048576 untimed 97.4 ms is 1629% of the 6.0 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | minimum_strided_read_f64_n1048576 untimed 83.8 ms is 1422% of the 5.9 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | pytorch-cuda | abs_strided_read_f64_n1048576 untimed 82.7 ms is 1716% of the 4.8 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | pytorch-cuda | sign_strided_read_f64_n1048576 untimed 70.3 ms is 1860% of the 3.8 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | expm1_strided_read_f64_n1048576 untimed 79.6 ms is 1682% of the 4.7 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | pytorch-cuda | log_strided_read_f64_n1048576 untimed 85.8 ms is 1415% of the 6.1 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | log1p_strided_read_f64_n1048576 untimed 70.8 ms is 1121% of the 6.3 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | pytorch-cuda | erf_strided_read_f64_n1048576 untimed 89.7 ms is 1301% of the 6.9 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | pytorch-cuda | sin_strided_read_f64_n1048576 untimed 84.9 ms is 1577% of the 5.4 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | pytorch-cuda | cos_strided_read_f64_n1048576 untimed 80.5 ms is 1476% of the 5.5 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | sqrt_strided_read_f64_n1048576 untimed 74.6 ms is 1954% of the 3.8 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | rsqrt_strided_read_f64_n1048576 untimed 69.0 ms is 1802% of the 3.8 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | reduce_prod_strided_read_f64_n1048576 untimed 734.9 ms is 1388% of the 53.0 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | reduce_max_strided_read_f64_n1048576 untimed 27.6 ms is 53% of the 51.9 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | reduce_min_strided_read_f64_n1048576 untimed 20.6 ms is 39% of the 52.4 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_squares_strided_read_f64_n1048576 untimed 26.1 ms is 64% of the 40.9 ms timed region (budget 20%) |
| tensordot_f64_n262144 | jax-cuda | tensordot_f64_n262144 untimed 703.1 ms is 1356% of the 51.8 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | jax-cuda | vdot_read_f64_n262144 untimed 171.1 ms is 466% of the 36.7 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | jax-cuda | norm_squared_read_f64_n262144 untimed 12.2 ms is 34% of the 36.2 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | jax-cuda | add_strided_read_f64_n1048576 untimed 341.1 ms is 1706% of the 20.0 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | jax-cuda | mul_strided_read_f64_n1048576 untimed 215.3 ms is 555% of the 38.8 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | jax-cuda | neg_strided_read_f64_n1048576 untimed 192.2 ms is 501% of the 38.4 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | jax-cuda | exp_strided_read_f64_n1048576 untimed 200.1 ms is 716% of the 27.9 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | jax-cuda | tanh_strided_read_f64_n1048576 untimed 211.0 ms is 886% of the 23.8 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | reduce_sum_strided_read_f64_n1048576 untimed 183.9 ms is 471% of the 39.0 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | jax-cuda | conj_strided_read_c64_n1048576 untimed 359.6 ms is 3079% of the 11.7 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | jax-cuda | dot_general_batch_f64_s256 untimed 586.2 ms is 1434% of the 40.9 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | jax-cuda | dot_conj_l0r0_c64_n262144 untimed 81.3 ms is 159% of the 51.1 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | jax-cuda | einsum_plan_owned_f64_s256 untimed 450.0 ms is 908% of the 49.6 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | jax-cuda | einsum_plan_read_f64_s256 untimed 13.5 ms is 29% of the 46.4 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | jax-cuda | einsum_plan_strided_read_f64_s256 untimed 215.1 ms is 457% of the 47.1 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | jax-cuda | sub_strided_read_f64_n1048576 untimed 191.1 ms is 681% of the 28.1 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | jax-cuda | div_strided_read_f64_n1048576 untimed 184.6 ms is 776% of the 23.8 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | jax-cuda | rem_strided_read_f64_n1048576 untimed 241.3 ms is 1055% of the 22.9 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | jax-cuda | pow_strided_read_f64_n1048576 untimed 233.3 ms is 823% of the 28.3 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | jax-cuda | maximum_strided_read_f64_n1048576 untimed 178.5 ms is 622% of the 28.7 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | jax-cuda | minimum_strided_read_f64_n1048576 untimed 200.9 ms is 636% of the 31.6 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | jax-cuda | abs_strided_read_f64_n1048576 untimed 175.4 ms is 652% of the 26.9 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | jax-cuda | sign_strided_read_f64_n1048576 untimed 190.2 ms is 630% of the 30.2 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | jax-cuda | expm1_strided_read_f64_n1048576 untimed 232.6 ms is 648% of the 35.9 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | jax-cuda | log_strided_read_f64_n1048576 untimed 221.3 ms is 768% of the 28.8 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | jax-cuda | log1p_strided_read_f64_n1048576 untimed 242.6 ms is 689% of the 35.2 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | jax-cuda | erf_strided_read_f64_n1048576 untimed 226.9 ms is 499% of the 45.5 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | jax-cuda | sin_strided_read_f64_n1048576 untimed 251.0 ms is 811% of the 30.9 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | jax-cuda | cos_strided_read_f64_n1048576 untimed 213.2 ms is 743% of the 28.7 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | jax-cuda | sqrt_strided_read_f64_n1048576 untimed 188.8 ms is 645% of the 29.3 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | rsqrt_strided_read_f64_n1048576 untimed 208.3 ms is 471% of the 44.2 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | reduce_prod_strided_read_f64_n1048576 untimed 183.9 ms is 549% of the 33.5 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | reduce_max_strided_read_f64_n1048576 untimed 207.7 ms is 643% of the 32.3 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | reduce_min_strided_read_f64_n1048576 untimed 102.6 ms is 343% of the 29.9 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | reduce_sum_squares_strided_read_f64_n1048576 untimed 221.0 ms is 597% of the 37.0 ms timed region (budget 20%) |