# gpu/reuse_contraction: CUDA public API benchmark

Measured cells: **150**. Unsupported cells: **74**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1410.6 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261107_201728/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**29** of 56) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Transposed borrowed-read cases use prepared native CUDA views in tenferro and PyTorch. JAX compares the same logical values in its native representation; its columns do not claim an equivalent public strided-view contract. View creation and input conversion are outside timing.

Generalized batch dot preserves native output order: tenferro [m,n,b], PyTorch/JAX [b,m,n]. Input permutation views and dimension-number configurations are prepared before timing. Output axes are normalized only on downloaded CPU arrays for verification; no timed output transpose or copy is added.

Prepared einsum uses the public ConcreteEinsumPlan for ik,kj->ij, built once outside execution and sampling. Native matrix multiplication/reuse/accumulation references match the selected mathematical and output-ownership contracts; they do not assert an equivalent prepared-plan interface. Transposed input views/values, read/write handles and accumulation seed restoration are untimed.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| add_into_f64_n1048576 | 26.154 (0.161) | unsupported | 12.023 (0.042) | unsupported |
| add_read_into_f64_n1048576 | 26.185 (0.075) | unsupported | 11.952 (0.047) | unsupported |
| sub_into_f64_n1048576 | 26.194 (0.162) | unsupported | 11.952 (0.060) | unsupported |
| sub_read_into_f64_n1048576 | 26.124 (0.174) | unsupported | 11.885 (0.065) | unsupported |
| mul_into_f64_n1048576 | 26.123 (0.162) | unsupported | 12.016 (0.062) | unsupported |
| mul_read_into_f64_n1048576 | 26.110 (0.118) | unsupported | 12.012 (0.027) | unsupported |
| div_into_f64_n1048576 | 26.259 (0.111) | unsupported | 13.126 (0.095) | unsupported |
| div_read_into_f64_n1048576 | 26.242 (0.056) | unsupported | 13.043 (0.124) | unsupported |
| neg_into_f64_n1048576 | 23.650 (0.176) | unsupported | 9.502 (0.068) | unsupported |
| neg_read_into_f64_n1048576 | 23.680 (0.088) | unsupported | 9.628 (0.060) | unsupported |
| conj_into_c64_n1048576 | 37.378 (0.444) | unsupported | 25.056 (0.354) | unsupported |
| conj_read_into_c64_n1048576 | 37.502 (0.345) | unsupported | 25.124 (0.208) | unsupported |
| tensordot_f64_n262144 † | 45.901 (0.191) | 49.033 (2.879) | 30.656 (0.046) | 129.555 (3.820) |
| vdot_read_f64_n262144 † | 80.661 (3.237) | unsupported | 12.368 (0.078) | 109.138 (1.249) |
| norm_squared_read_f64_n262144 † | 29.535 (0.127) | unsupported | 12.487 (0.086) | 86.177 (27.944) |
| axpby_read_into_accum_f64_n1048576 | 38.364 (0.588) | unsupported | 16.919 (0.261) | unsupported |
| dot_general_read_into_f64_s256 † | 20.778 (0.954) | unsupported | 11.927 (0.039) | unsupported |
| dot_general_read_into_accum_f64_s256 † | 22.252 (0.786) | unsupported | 13.218 (0.097) | unsupported |
| dot_general_read_into_accum_cached_f64_s256 † | 24.339 (0.127) | unsupported | 13.286 (0.142) | unsupported |
| add_strided_read_f64_n1048576 † | 135.147 (1.457) | unsupported | 12.260 (0.036) | 57.954 (1.952) |
| mul_strided_read_f64_n1048576 † | 134.746 (0.550) | unsupported | 12.220 (0.057) | 57.968 (2.933) |
| neg_strided_read_f64_n1048576 | 81.254 (0.473) | unsupported | 9.646 (0.162) | 55.630 (1.448) |
| exp_strided_read_f64_n1048576 | 82.646 (0.797) | unsupported | 10.005 (0.057) | 51.457 (1.989) |
| tanh_strided_read_f64_n1048576 | 82.334 (0.255) | unsupported | 10.007 (0.069) | 49.581 (1.746) |
| reduce_sum_strided_read_f64_n1048576 † | 100.929 (1.281) | unsupported | 16.947 (0.015) | 53.019 (0.653) |
| conj_strided_read_c64_n1048576 | 112.336 (1.937) | unsupported | 25.266 (0.288) | 57.621 (3.172) |
| dot_general_batch_f64_s256 † | 31.611 (0.226) | 32.101 (0.133) | 25.379 (0.179) | 78.348 (1.088) |
| dot_conj_l0r0_c64_n262144 † | 43.798 (1.202) | 39.768 (0.415) | 15.457 (0.013) | 89.406 (26.384) |
| einsum_plan_owned_f64_s256 † | 27.704 (0.283) | unsupported | 12.889 (0.085) | 50.546 (0.953) |
| einsum_plan_read_f64_s256 † | 27.602 (0.067) | unsupported | 12.935 (0.057) | 51.279 (0.947) |
| einsum_plan_strided_read_f64_s256 † | 83.878 (1.909) | unsupported | 13.356 (0.176) | 51.200 (1.131) |
| einsum_plan_into_f64_s256 † | 20.231 (0.205) | unsupported | 10.763 (0.411) | unsupported |
| einsum_plan_read_into_f64_s256 † | 20.847 (0.132) | unsupported | 10.696 (0.053) | unsupported |
| einsum_plan_read_into_accum_f64_s256 † | 22.341 (0.122) | unsupported | 12.974 (0.203) | unsupported |
| einsum_plan_strided_read_into_f64_s256 † | 78.547 (4.787) | unsupported | 10.925 (0.045) | unsupported |
| einsum_plan_strided_read_into_accum_f64_s256 † | 79.448 (1.475) | unsupported | 13.406 (0.112) | unsupported |
| sub_strided_read_f64_n1048576 † | 134.489 (3.418) | unsupported | 12.244 (0.094) | 57.854 (2.294) |
| div_strided_read_f64_n1048576 † | 134.750 (0.392) | unsupported | 13.099 (0.068) | 52.307 (2.233) |
| rem_strided_read_f64_n1048576 † | 171.158 (0.818) | unsupported | 12.171 (0.156) | 50.593 (2.820) |
| pow_strided_read_f64_n1048576 † | 170.703 (1.418) | unsupported | 35.156 (0.086) | 59.746 (1.846) |
| maximum_strided_read_f64_n1048576 † | 154.261 (2.818) | unsupported | 12.183 (0.078) | 56.565 (1.177) |
| minimum_strided_read_f64_n1048576 † | 153.194 (1.134) | unsupported | 12.101 (0.082) | 58.351 (2.087) |
| abs_strided_read_f64_n1048576 | 81.400 (0.497) | unsupported | 10.596 (0.268) | 50.793 (1.324) |
| sign_strided_read_f64_n1048576 | 81.671 (0.312) | unsupported | 9.565 (0.053) | 49.376 (1.048) |
| expm1_strided_read_f64_n1048576 | 83.693 (0.675) | unsupported | 11.916 (0.059) | 48.705 (1.599) |
| log_strided_read_f64_n1048576 | 87.222 (0.245) | unsupported | 15.226 (0.061) | 49.240 (1.384) |
| log1p_strided_read_f64_n1048576 | 87.029 (0.362) | unsupported | 15.867 (0.043) | 47.845 (1.081) |
| erf_strided_read_f64_n1048576 | 89.170 (0.248) | unsupported | 17.521 (0.053) | 64.325 (1.750) |
| sin_strided_read_f64_n1048576 | 85.170 (0.400) | unsupported | 13.526 (0.059) | 50.274 (0.777) |
| cos_strided_read_f64_n1048576 | 85.409 (0.144) | unsupported | 13.601 (0.046) | 50.611 (1.505) |
| sqrt_strided_read_f64_n1048576 | 81.906 (0.288) | unsupported | 9.649 (0.105) | 50.165 (3.210) |
| rsqrt_strided_read_f64_n1048576 | 81.861 (0.473) | unsupported | 9.511 (0.038) | 69.964 (3.117) |
| reduce_prod_strided_read_f64_n1048576 † | 100.380 (0.543) | unsupported | 16.953 (0.027) | 54.494 (0.568) |
| reduce_max_strided_read_f64_n1048576 † | 100.983 (0.486) | unsupported | 17.058 (0.019) | 106.541 (1.230) |
| reduce_min_strided_read_f64_n1048576 † | 93.390 (1.304) | unsupported | 17.106 (0.025) | 107.862 (2.528) |
| reduce_sum_squares_strided_read_f64_n1048576 † | 92.716 (0.289) | unsupported | 23.198 (0.166) | 1814.296 (27.862) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.625 | 16777216 | 0.234 | 0 |
| abs_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.339 | 16777216 | 1.122 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.605 | 16777216 | 0.146 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| add_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.385 | 25165824 | 1.484 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.837 | 25165824 | 0.682 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.382 | 25165824 | 1.493 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.838 | 25165824 | 0.681 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.855 | 25165824 | 0.308 | 0 |
| add_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.392 | 25165824 | 1.455 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.162 | 25165824 | 0.132 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| axpby_read_into_accum_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.541 | 25165824 | 1.054 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.228 | 25165824 | 0.465 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No matching public caller-owned accumulating output API; donation is not output reuse. |
| conj_into_c64_n1048576 | jax-cuda | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.401 | 33554432 | 0.949 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.598 | 33554432 | 0.636 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1048576 | jax-cuda | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.402 | 33554432 | 0.947 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.600 | 33554432 | 0.634 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_strided_read_c64_n1048576 | jax-cuda | 16 | 15 | 0.922 | 33554432 | 0.413 | 0 |
| conj_strided_read_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.404 | 33554432 | 0.941 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.797 | 33554432 | 0.212 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| cos_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.620 | 16777216 | 0.235 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.435 | 16777216 | 0.874 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.733 | 16777216 | 0.139 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| div_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.420 | 25165824 | 1.359 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.840 | 25165824 | 0.679 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.417 | 25165824 | 1.368 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.840 | 25165824 | 0.680 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.674 | 25165824 | 0.341 | 0 |
| div_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.419 | 25165824 | 1.362 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.156 | 25165824 | 0.132 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| dot_conj_l0r0_c64_n262144 | jax-cuda | 64 | 15 | 5.722 | 8388624 | 0.067 | 1.221e-05 |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | 256 | 15 | 3.957 | 8388624 | 0.385 | 1.366e-05 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | 64 | 15 | 2.803 | 8388624 | 0.136 | 1.212e-05 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | 64 | 15 | 2.545 | 8388624 | 0.150 | 1.212e-05 |
| dot_general_batch_f64_s256 | jax-cuda | 32 | 15 | 2.507 | 3210240 | 0.029 | 7.463e-07 |
| dot_general_batch_f64_s256 | pytorch-cuda | 128 | 15 | 3.248 | 3210240 | 0.090 | 7.463e-07 |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | 64 | 15 | 2.023 | 3210240 | 0.072 | 7.463e-07 |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | 64 | 15 | 2.054 | 3210240 | 0.071 | 7.463e-07 |
| dot_general_read_into_accum_cached_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | 256 | 15 | 3.401 | 2097152 | 0.112 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | 128 | 15 | 3.115 | 2097152 | 0.061 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 3.384 | 2097152 | 0.112 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.848 | 2097152 | 0.067 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 3.053 | 2097152 | 0.125 | 0 |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.660 | 2097152 | 0.072 | 0 |
| dot_general_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| einsum_plan_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.755 | 1070080 | 0.070 | 0 |
| einsum_plan_into_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.590 | 1070080 | 0.037 | 0 |
| einsum_plan_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_owned_f64_s256 | jax-cuda | 64 | 15 | 3.235 | 1070080 | 0.015 | 0 |
| einsum_plan_owned_f64_s256 | pytorch-cuda | 256 | 15 | 3.300 | 1070080 | 0.059 | 0 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-direct | 128 | 15 | 3.546 | 1070080 | 0.027 | 0 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_f64_s256 | jax-cuda | 64 | 15 | 3.282 | 1070080 | 0.015 | 0 |
| einsum_plan_read_f64_s256 | pytorch-cuda | 256 | 15 | 3.311 | 1070080 | 0.059 | 0 |
| einsum_plan_read_f64_s256 | tenferro-cuda-direct | 128 | 15 | 3.533 | 1070080 | 0.027 | 0 |
| einsum_plan_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_accum_f64_s256 | jax-cuda | — | — | — | 1608704 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 3.321 | 1608704 | 0.088 | 0 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.860 | 1608704 | 0.051 | 0 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.738 | 1070080 | 0.071 | 0 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.668 | 1070080 | 0.036 | 0 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_f64_s256 | jax-cuda | 64 | 15 | 3.277 | 1070080 | 0.015 | 0 |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | 256 | 15 | 3.419 | 1070080 | 0.057 | 0 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-direct | 32 | 15 | 2.684 | 1070080 | 0.009 | 0 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_accum_f64_s256 | jax-cuda | — | — | — | 1608704 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 3.432 | 1608704 | 0.085 | 0 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-direct | 32 | 15 | 2.542 | 1608704 | 0.014 | 0 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.797 | 1070080 | 0.069 | 0 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-direct | 32 | 15 | 2.514 | 1070080 | 0.010 | 0 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| erf_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.058 | 16777216 | 0.185 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.561 | 16777216 | 0.679 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.853 | 16777216 | 0.133 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| exp_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.647 | 16777216 | 0.231 | 1.095e-07 |
| exp_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.320 | 16777216 | 1.189 | 1.095e-07 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.645 | 16777216 | 0.144 | 1.095e-07 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| expm1_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.559 | 16777216 | 0.244 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.381 | 16777216 | 0.998 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.678 | 16777216 | 0.142 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log1p_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.531 | 16777216 | 0.249 | 1.11e-07 |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.508 | 16777216 | 0.750 | 1.11e-07 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.785 | 16777216 | 0.137 | 1.11e-07 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.576 | 16777216 | 0.242 | 4.308e-08 |
| log_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.487 | 16777216 | 0.781 | 4.308e-08 |
| log_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.791 | 16777216 | 0.136 | 4.308e-08 |
| log_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| maximum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.810 | 25165824 | 0.315 | 0 |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.390 | 25165824 | 1.464 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.468 | 25165824 | 0.116 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| minimum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.867 | 25165824 | 0.306 | 0 |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.387 | 25165824 | 1.474 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.451 | 25165824 | 0.116 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| mul_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.385 | 25165824 | 1.485 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.836 | 25165824 | 0.683 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.384 | 25165824 | 1.485 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.836 | 25165824 | 0.683 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.855 | 25165824 | 0.308 | 0 |
| mul_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.391 | 25165824 | 1.460 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.156 | 25165824 | 0.132 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| neg_into_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.304 | 16777216 | 1.252 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.757 | 16777216 | 0.503 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.308 | 16777216 | 1.235 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.758 | 16777216 | 0.502 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.780 | 16777216 | 0.214 | 0 |
| neg_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.309 | 16777216 | 1.233 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.600 | 16777216 | 0.146 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| norm_squared_read_f64_n262144 | jax-cuda | 32 | 15 | 2.758 | 2097160 | 0.017 | 1.664e-07 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 256 | 15 | 3.197 | 2097160 | 0.119 | 1.664e-07 |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | 128 | 15 | 3.780 | 2097160 | 0.050 | 1.664e-07 |
| norm_squared_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 2097160 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| pow_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.912 | 25165824 | 0.299 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.125 | 25165824 | 0.507 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.731 | 25165824 | 0.105 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 3.409 | 8392704 | 0.056 | 0 |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.183 | 8392704 | 0.349 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.231 | 8392704 | 0.059 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 3.452 | 8392704 | 0.055 | 0 |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.190 | 8392704 | 0.348 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.988 | 8392704 | 0.064 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 3.488 | 8392704 | 0.109 | 4.995e-06 |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.170 | 8392704 | 0.351 | 4.106e-06 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.212 | 8392704 | 0.059 | 4.661e-06 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | 2 | 15 | 3.629 | 8392704 | 0.003 | 3.311e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.969 | 8392704 | 0.256 | 3.309e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.967 | 8392704 | 0.064 | 3.312e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 3.393 | 8392704 | 0.112 | 7.159e-06 |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.169 | 8392704 | 0.351 | 5.935e-06 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.230 | 8392704 | 0.059 | 4.724e-06 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rem_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.619 | 25165824 | 0.353 | 0 |
| rem_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.389 | 25165824 | 1.466 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.739 | 25165824 | 0.104 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.239 | 16777216 | 0.170 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.304 | 16777216 | 1.251 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.620 | 16777216 | 0.145 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sign_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.580 | 16777216 | 0.241 | 0 |
| sign_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.306 | 16777216 | 1.243 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.613 | 16777216 | 0.146 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sin_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.609 | 16777216 | 0.237 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.433 | 16777216 | 0.879 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.725 | 16777216 | 0.140 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sqrt_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.605 | 16777216 | 0.237 | 0 |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.309 | 16777216 | 1.233 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.621 | 16777216 | 0.145 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sub_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.382 | 25165824 | 1.493 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.838 | 25165824 | 0.681 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.380 | 25165824 | 1.501 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.836 | 25165824 | 0.683 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.851 | 25165824 | 0.308 | 0 |
| sub_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.392 | 25165824 | 1.457 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.152 | 25165824 | 0.133 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tanh_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.587 | 16777216 | 0.240 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.320 | 16777216 | 1.189 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.635 | 16777216 | 0.144 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tensordot_f64_n262144 | jax-cuda | 16 | 15 | 2.073 | 6291456 | 0.034 | 6.714e-06 |
| tensordot_f64_n262144 | pytorch-cuda | 128 | 15 | 3.924 | 6291456 | 0.145 | 6.714e-06 |
| tensordot_f64_n262144 | tenferro-cuda-direct | 64 | 15 | 2.938 | 6291456 | 0.097 | 6.714e-06 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 32 | 15 | 1.569 | 6291456 | 0.091 | 6.714e-06 |
| vdot_read_f64_n262144 | jax-cuda | 32 | 15 | 3.492 | 4194312 | 0.027 | 2.995e-06 |
| vdot_read_f64_n262144 | pytorch-cuda | 256 | 15 | 3.166 | 4194312 | 0.240 | 2.995e-06 |
| vdot_read_f64_n262144 | tenferro-cuda-direct | 32 | 15 | 2.581 | 4194312 | 0.037 | 2.995e-06 |
| vdot_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 4194312 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261107_201728/gpu_reuse_contraction.yaml
timestamp: '2026-11-07T20:17:28+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: ffd352bdb9f98537be747101bc278db8a50a1595
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: e7860ca369657a92a929026378d2a4c11adaae16
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
  cpu: AMD EPYC 7742 64-Core Processor
  logical_cpus: 256
  power:
    cpufreq_boost: '1'
    governor: schedutil
    scaling_max_freq_khz: '2250000'
    smt:
      active: '1'
      control: 'on'
  numa:
    nodes: 8
    cpulists:
      '0': 0-15,128-143
      '1': 16-31,144-159
      '2': 32-47,160-175
      '3': 48-63,176-191
      '4': 64-79,192-207
      '5': 80-95,208-223
      '6': 96-111,224-239
      '7': 112-127,240-255
  smt_siblings:
  - 0,128
  - 1,129
  - 2,130
  - 3,131
  - 4,132
  - 5,133
  - 6,134
  - 7,135
  - 8,136
  - 9,137
  - 10,138
  - 11,139
  - 12,140
  - 13,141
  - 14,142
  - 15,143
  - 16,144
  - 17,145
  - 18,146
  - 19,147
  - 20,148
  - 21,149
  - 22,150
  - 23,151
  - 24,152
  - 25,153
  - 26,154
  - 27,155
  - 28,156
  - 29,157
  - 30,158
  - 31,159
  - 32,160
  - 33,161
  - 34,162
  - 35,163
  - 36,164
  - 37,165
  - 38,166
  - 39,167
  - 40,168
  - 41,169
  - 42,170
  - 43,171
  - 44,172
  - 45,173
  - 46,174
  - 47,175
  - 48,176
  - 49,177
  - 50,178
  - 51,179
  - 52,180
  - 53,181
  - 54,182
  - 55,183
  - 56,184
  - 57,185
  - 58,186
  - 59,187
  - 60,188
  - 61,189
  - 62,190
  - 63,191
  - 64,192
  - 65,193
  - 66,194
  - 67,195
  - 68,196
  - 69,197
  - 70,198
  - 71,199
  - 72,200
  - 73,201
  - 74,202
  - 75,203
  - 76,204
  - 77,205
  - 78,206
  - 79,207
  - 80,208
  - 81,209
  - 82,210
  - 83,211
  - 84,212
  - 85,213
  - 86,214
  - 87,215
  - 88,216
  - 89,217
  - 90,218
  - 91,219
  - 92,220
  - 93,221
  - 94,222
  - 95,223
  - 96,224
  - 97,225
  - 98,226
  - 99,227
  - 100,228
  - 101,229
  - 102,230
  - 103,231
  - 104,232
  - 105,233
  - 106,234
  - 107,235
  - 108,236
  - 109,237
  - 110,238
  - 111,239
  - 112,240
  - 113,241
  - 114,242
  - 115,243
  - 116,244
  - 117,245
  - 118,246
  - 119,247
  - 120,248
  - 121,249
  - 122,250
  - 123,251
  - 124,252
  - 125,253
  - 126,254
  - 127,255
  l3_domains:
  - 0-3,128-131
  - 4-7,132-135
  - 8-11,136-139
  - 12-15,140-143
  - 16-19,144-147
  - 20-23,148-151
  - 24-27,152-155
  - 28-31,156-159
  - 32-35,160-163
  - 36-39,164-167
  - 40-43,168-171
  - 44-47,172-175
  - 48-51,176-179
  - 52-55,180-183
  - 56-59,184-187
  - 60-63,188-191
  - 64-67,192-195
  - 68-71,196-199
  - 72-75,200-203
  - 76-79,204-207
  - 80-83,208-211
  - 84-87,212-215
  - 88-91,216-219
  - 92-95,220-223
  - 96-99,224-227
  - 100-103,228-231
  - 104-107,232-235
  - 108-111,236-239
  - 112-115,240-243
  - 116-119,244-247
  - 120-123,248-251
  - 124-127,252-255
build:
  toolchain:
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: 86889f3412c3
  os: Linux-6.8.0-138-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-255
    l3_domains:
    - 0-3,128-131
    - 4-7,132-135
    - 8-11,136-139
    - 12-15,140-143
    - 16-19,144-147
    - 20-23,148-151
    - 24-27,152-155
    - 28-31,156-159
    - 32-35,160-163
    - 36-39,164-167
    - 40-43,168-171
    - 44-47,172-175
    - 48-51,176-179
    - 52-55,180-183
    - 56-59,184-187
    - 60-63,188-191
    - 64-67,192-195
    - 68-71,196-199
    - 72-75,200-203
    - 76-79,204-207
    - 80-83,208-211
    - 84-87,212-215
    - 88-91,216-219
    - 92-95,220-223
    - 96-99,224-227
    - 100-103,228-231
    - 104-107,232-235
    - 108-111,236-239
    - 112-115,240-243
    - 116-119,244-247
    - 120-123,248-251
    - 124-127,252-255
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
    GPU_RUNTIME_SHA256: ecc740128a444e2b7d8a80ea018e5fbeb2da956dc2669f05886d8d7c97b49c67
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
  device_name: NVIDIA A100-SXM4-80GB
  device_uuid: GPU-3f509591-db26-6474-253e-fcb29d2b374b
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
  device_name: NVIDIA A100-SXM4-80GB
  buffer_bytes: 67108864
  traffic_bytes_per_copy: 134217728
  traffic_model: one read plus one write of buffer_bytes per copy
  warmups: 3
  sample_ms:
  - 0.096152
  - 0.09492
  - 0.092685
  - 0.098286
  - 0.096754
  - 0.093968
  - 0.09515
  - 0.097555
  - 0.097594
  - 0.092705
  - 0.094639
  - 0.097034
  - 0.095701
  - 0.094539
  - 0.094529
  median_ms: 0.09515
  bytes_per_second: 1410590940620.0737
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-07T20:37:47.261982+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-3f509591-db26-6474-253e-fcb29d2b374b
  cuda_visible_devices: GPU-3f509591-db26-6474-253e-fcb29d2b374b
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
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 1919e33cc5f2d0208901837f80cb7c0600426462f998be6dd933062d9f6bb345
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 788b9fde4f8ca4162921baef1203b2e31bb747ec30327a425de3442cf872df52
    data/results/nvidia-gpu/gpu/reuse_contraction/20261107_201728/fixtures/manifest.json: 706a6a5d725bf440a03748d8ba95eeff194de50a81b8547ec97f56312b6629ef
    target/release/benchmark_gpu_public_api: 94cf356a747431a654002f07d7e16069b1849c82e215e2fe5d75a94346f4ae17
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-07T20:40:19Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-07T20:39:35Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-07T20:38:34Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-07T20:38:38Z'
  compile:
    case_wall_seconds: 115.176
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
| add_into_f64_n1048576 | tenferro-cuda-direct | add_into_f64_n1048576 untimed 520.3 ms is 4142% of the 12.6 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | add_read_into_f64_n1048576 untimed 399.5 ms is 3180% of the 12.6 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | tenferro-cuda-direct | sub_into_f64_n1048576 untimed 397.4 ms is 3163% of the 12.6 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | sub_read_into_f64_n1048576 untimed 376.3 ms is 2996% of the 12.6 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | tenferro-cuda-direct | mul_into_f64_n1048576 untimed 385.3 ms is 3068% of the 12.6 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | mul_read_into_f64_n1048576 untimed 369.1 ms is 2945% of the 12.5 ms timed region (budget 20%) |
| div_into_f64_n1048576 | tenferro-cuda-direct | div_into_f64_n1048576 untimed 361.6 ms is 2835% of the 12.8 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | div_read_into_f64_n1048576 untimed 377.5 ms is 2997% of the 12.6 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | tenferro-cuda-direct | neg_into_f64_n1048576 untimed 383.0 ms is 3373% of the 11.4 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | neg_read_into_f64_n1048576 untimed 384.8 ms is 3358% of the 11.5 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | tenferro-cuda-direct | conj_into_c64_n1048576 untimed 414.3 ms is 4584% of the 9.0 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | conj_read_into_c64_n1048576 untimed 424.2 ms is 4718% of the 9.0 ms timed region (budget 20%) |
| tensordot_f64_n262144 | tenferro-cuda-direct | tensordot_f64_n262144 untimed 2813.2 ms is 5433% of the 51.8 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | tenferro-cuda-direct | vdot_read_f64_n262144 untimed 12.3 ms is 32% of the 38.9 ms timed region (budget 20%) |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | axpby_read_into_accum_f64_n1048576 untimed 390.3 ms is 2121% of the 18.4 ms timed region (budget 20%) |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | dot_general_read_into_f64_s256 untimed 291.3 ms is 725% of the 40.2 ms timed region (budget 20%) |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | dot_general_read_into_accum_f64_s256 untimed 377.1 ms is 866% of the 43.6 ms timed region (budget 20%) |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | dot_general_read_into_accum_cached_f64_s256 untimed 386.4 ms is 826% of the 46.8 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | add_strided_read_f64_n1048576 untimed 129.3 ms is 398% of the 32.5 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | mul_strided_read_f64_n1048576 untimed 140.1 ms is 431% of the 32.5 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | neg_strided_read_f64_n1048576 untimed 115.0 ms is 295% of the 39.1 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | exp_strided_read_f64_n1048576 untimed 116.2 ms is 291% of the 39.9 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | tanh_strided_read_f64_n1048576 untimed 121.0 ms is 306% of the 39.5 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_sum_strided_read_f64_n1048576 untimed 29.4 ms is 60% of the 48.8 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | conj_strided_read_c64_n1048576 untimed 157.1 ms is 578% of the 27.2 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | dot_general_batch_f64_s256 untimed 35.8 ms is 118% of the 30.3 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | dot_conj_l0r0_c64_n262144 untimed 23.9 ms is 57% of the 42.2 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | tenferro-cuda-direct | einsum_plan_owned_f64_s256 untimed 10.8 ms is 20% of the 53.5 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | tenferro-cuda-direct | einsum_plan_read_f64_s256 untimed 10.9 ms is 21% of the 53.1 ms timed region (budget 20%) |
| einsum_plan_into_f64_s256 | tenferro-cuda-direct | einsum_plan_into_f64_s256 untimed 523.6 ms is 1342% of the 39.0 ms timed region (budget 20%) |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-direct | einsum_plan_read_into_f64_s256 untimed 538.9 ms is 1346% of the 40.0 ms timed region (budget 20%) |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-direct | einsum_plan_read_into_accum_f64_s256 untimed 578.0 ms is 1342% of the 43.1 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_into_f64_s256 untimed 510.8 ms is 1304% of the 39.2 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_into_accum_f64_s256 untimed 532.8 ms is 1384% of the 38.5 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | tenferro-cuda-direct | sub_strided_read_f64_n1048576 untimed 134.1 ms is 411% of the 32.6 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | tenferro-cuda-direct | div_strided_read_f64_n1048576 untimed 134.8 ms is 417% of the 32.4 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | tenferro-cuda-direct | rem_strided_read_f64_n1048576 untimed 132.7 ms is 321% of the 41.4 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | tenferro-cuda-direct | pow_strided_read_f64_n1048576 untimed 140.0 ms is 340% of the 41.2 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-direct | maximum_strided_read_f64_n1048576 untimed 133.4 ms is 360% of the 37.0 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-direct | minimum_strided_read_f64_n1048576 untimed 132.9 ms is 360% of the 37.0 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | tenferro-cuda-direct | abs_strided_read_f64_n1048576 untimed 118.3 ms is 302% of the 39.1 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | tenferro-cuda-direct | sign_strided_read_f64_n1048576 untimed 115.1 ms is 293% of the 39.3 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-direct | expm1_strided_read_f64_n1048576 untimed 116.0 ms is 287% of the 40.4 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | tenferro-cuda-direct | log_strided_read_f64_n1048576 untimed 117.2 ms is 280% of the 41.9 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-direct | log1p_strided_read_f64_n1048576 untimed 119.7 ms is 286% of the 41.8 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | tenferro-cuda-direct | erf_strided_read_f64_n1048576 untimed 127.5 ms is 298% of the 42.8 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | tenferro-cuda-direct | sin_strided_read_f64_n1048576 untimed 122.7 ms is 299% of the 41.0 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | tenferro-cuda-direct | cos_strided_read_f64_n1048576 untimed 115.6 ms is 281% of the 41.2 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | sqrt_strided_read_f64_n1048576 untimed 116.6 ms is 296% of the 39.4 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | rsqrt_strided_read_f64_n1048576 untimed 123.4 ms is 311% of the 39.7 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_prod_strided_read_f64_n1048576 untimed 29.7 ms is 61% of the 48.5 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_max_strided_read_f64_n1048576 untimed 21.8 ms is 45% of the 48.4 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_min_strided_read_f64_n1048576 untimed 21.1 ms is 47% of the 45.3 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_sum_squares_strided_read_f64_n1048576 untimed 21.4 ms is 48% of the 44.6 ms timed region (budget 20%) |
| tensordot_f64_n262144 | tenferro-cuda-eager | tensordot_f64_n262144 untimed 3003.0 ms is 12067% of the 24.9 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | dot_general_batch_f64_s256 untimed 14.5 ms is 47% of the 30.9 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | dot_conj_l0r0_c64_n262144 untimed 17.8 ms is 46% of the 38.4 ms timed region (budget 20%) |
| add_into_f64_n1048576 | pytorch-cuda | add_into_f64_n1048576 untimed 1658.5 ms is 28717% of the 5.8 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | pytorch-cuda | add_read_into_f64_n1048576 untimed 1404.1 ms is 24433% of the 5.7 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | pytorch-cuda | sub_into_f64_n1048576 untimed 1402.8 ms is 24462% of the 5.7 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | pytorch-cuda | sub_read_into_f64_n1048576 untimed 1352.8 ms is 23505% of the 5.8 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | pytorch-cuda | mul_into_f64_n1048576 untimed 1357.3 ms is 22776% of the 6.0 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | pytorch-cuda | mul_read_into_f64_n1048576 untimed 1397.7 ms is 24224% of the 5.8 ms timed region (budget 20%) |
| div_into_f64_n1048576 | pytorch-cuda | div_into_f64_n1048576 untimed 1364.1 ms is 21648% of the 6.3 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | pytorch-cuda | div_read_into_f64_n1048576 untimed 1382.8 ms is 22006% of the 6.3 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | pytorch-cuda | neg_into_f64_n1048576 untimed 1488.3 ms is 32646% of the 4.6 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | pytorch-cuda | neg_read_into_f64_n1048576 untimed 1358.3 ms is 29413% of the 4.6 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | pytorch-cuda | conj_into_c64_n1048576 untimed 1437.4 ms is 23838% of the 6.0 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | pytorch-cuda | conj_read_into_c64_n1048576 untimed 1394.7 ms is 23135% of the 6.0 ms timed region (budget 20%) |
| tensordot_f64_n262144 | pytorch-cuda | tensordot_f64_n262144 untimed 421.8 ms is 716% of the 58.9 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | pytorch-cuda | vdot_read_f64_n262144 untimed 13.5 ms is 28% of the 47.6 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | pytorch-cuda | norm_squared_read_f64_n262144 untimed 10.0 ms is 21% of the 48.1 ms timed region (budget 20%) |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | axpby_read_into_accum_f64_n1048576 untimed 1360.0 ms is 16723% of the 8.1 ms timed region (budget 20%) |
| dot_general_read_into_f64_s256 | pytorch-cuda | dot_general_read_into_f64_s256 untimed 154.9 ms is 338% of the 45.8 ms timed region (budget 20%) |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | dot_general_read_into_accum_f64_s256 untimed 187.6 ms is 367% of the 51.1 ms timed region (budget 20%) |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | dot_general_read_into_accum_cached_f64_s256 untimed 130.2 ms is 253% of the 51.5 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | pytorch-cuda | add_strided_read_f64_n1048576 untimed 1451.8 ms is 24574% of the 5.9 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | pytorch-cuda | mul_strided_read_f64_n1048576 untimed 1450.3 ms is 24694% of the 5.9 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | pytorch-cuda | neg_strided_read_f64_n1048576 untimed 1397.3 ms is 30073% of the 4.6 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | pytorch-cuda | exp_strided_read_f64_n1048576 untimed 1406.4 ms is 29269% of the 4.8 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | tanh_strided_read_f64_n1048576 untimed 1432.4 ms is 29655% of the 4.8 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_strided_read_f64_n1048576 untimed 47.0 ms is 144% of the 32.6 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | pytorch-cuda | conj_strided_read_c64_n1048576 untimed 1446.2 ms is 23806% of the 6.1 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | pytorch-cuda | dot_general_batch_f64_s256 untimed 284.8 ms is 585% of the 48.7 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | dot_conj_l0r0_c64_n262144 untimed 14.3 ms is 24% of the 59.3 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | pytorch-cuda | einsum_plan_owned_f64_s256 untimed 97.6 ms is 197% of the 49.5 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | pytorch-cuda | einsum_plan_read_f64_s256 untimed 98.0 ms is 197% of the 49.8 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | einsum_plan_strided_read_f64_s256 untimed 99.9 ms is 193% of the 51.8 ms timed region (budget 20%) |
| einsum_plan_into_f64_s256 | pytorch-cuda | einsum_plan_into_f64_s256 untimed 97.4 ms is 227% of the 42.9 ms timed region (budget 20%) |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | einsum_plan_read_into_f64_s256 untimed 178.1 ms is 431% of the 41.3 ms timed region (budget 20%) |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_read_into_accum_f64_s256 untimed 133.0 ms is 264% of the 50.3 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_f64_s256 untimed 96.1 ms is 229% of the 42.0 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_accum_f64_s256 untimed 133.0 ms is 259% of the 51.4 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | pytorch-cuda | sub_strided_read_f64_n1048576 untimed 1440.9 ms is 23740% of the 6.1 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | pytorch-cuda | div_strided_read_f64_n1048576 untimed 1449.0 ms is 23026% of the 6.3 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | pytorch-cuda | rem_strided_read_f64_n1048576 untimed 1459.9 ms is 24872% of the 5.9 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | pytorch-cuda | pow_strided_read_f64_n1048576 untimed 1504.2 ms is 8916% of the 16.9 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | maximum_strided_read_f64_n1048576 untimed 1478.2 ms is 25242% of the 5.9 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | minimum_strided_read_f64_n1048576 untimed 1466.6 ms is 25285% of the 5.8 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | pytorch-cuda | abs_strided_read_f64_n1048576 untimed 1410.3 ms is 27823% of the 5.1 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | pytorch-cuda | sign_strided_read_f64_n1048576 untimed 1414.4 ms is 30761% of the 4.6 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | expm1_strided_read_f64_n1048576 untimed 1440.7 ms is 25165% of the 5.7 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | pytorch-cuda | log_strided_read_f64_n1048576 untimed 1387.2 ms is 18960% of the 7.3 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | log1p_strided_read_f64_n1048576 untimed 1365.1 ms is 17930% of the 7.6 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | pytorch-cuda | erf_strided_read_f64_n1048576 untimed 1369.3 ms is 15922% of the 8.6 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | pytorch-cuda | sin_strided_read_f64_n1048576 untimed 1428.5 ms is 21988% of the 6.5 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | pytorch-cuda | cos_strided_read_f64_n1048576 untimed 1396.7 ms is 21396% of the 6.5 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | sqrt_strided_read_f64_n1048576 untimed 1372.2 ms is 29513% of the 4.6 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | rsqrt_strided_read_f64_n1048576 untimed 1360.3 ms is 29771% of the 4.6 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | reduce_prod_strided_read_f64_n1048576 untimed 931.6 ms is 2843% of the 32.8 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | reduce_max_strided_read_f64_n1048576 untimed 20.9 ms is 64% of the 32.8 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | reduce_min_strided_read_f64_n1048576 untimed 22.2 ms is 68% of the 32.8 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_squares_strided_read_f64_n1048576 untimed 69.9 ms is 157% of the 44.6 ms timed region (budget 20%) |
| tensordot_f64_n262144 | jax-cuda | tensordot_f64_n262144 untimed 1252.4 ms is 4030% of the 31.1 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | jax-cuda | vdot_read_f64_n262144 untimed 281.9 ms is 542% of the 52.0 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | jax-cuda | norm_squared_read_f64_n262144 untimed 10.5 ms is 29% of the 36.5 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | jax-cuda | add_strided_read_f64_n1048576 untimed 1803.1 ms is 6485% of the 27.8 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | jax-cuda | mul_strided_read_f64_n1048576 untimed 1581.9 ms is 5626% of the 28.1 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | jax-cuda | neg_strided_read_f64_n1048576 untimed 1562.5 ms is 5533% of the 28.2 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | jax-cuda | exp_strided_read_f64_n1048576 untimed 1571.8 ms is 6373% of the 24.7 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | jax-cuda | tanh_strided_read_f64_n1048576 untimed 1576.1 ms is 6117% of the 25.8 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | reduce_sum_strided_read_f64_n1048576 untimed 250.2 ms is 490% of the 51.1 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | jax-cuda | conj_strided_read_c64_n1048576 untimed 1751.5 ms is 12926% of the 13.6 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | jax-cuda | dot_general_batch_f64_s256 untimed 1042.5 ms is 2765% of the 37.7 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | jax-cuda | dot_conj_l0r0_c64_n262144 untimed 129.4 ms is 141% of the 91.5 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | jax-cuda | einsum_plan_owned_f64_s256 untimed 659.4 ms is 1342% of the 49.2 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | jax-cuda | einsum_plan_read_f64_s256 untimed 101.0 ms is 205% of the 49.3 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | jax-cuda | einsum_plan_strided_read_f64_s256 untimed 437.5 ms is 887% of the 49.3 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | jax-cuda | sub_strided_read_f64_n1048576 untimed 1567.4 ms is 5606% of the 28.0 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | jax-cuda | div_strided_read_f64_n1048576 untimed 1570.3 ms is 6218% of the 25.3 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | jax-cuda | rem_strided_read_f64_n1048576 untimed 1650.7 ms is 6678% of the 24.7 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | jax-cuda | pow_strided_read_f64_n1048576 untimed 1654.5 ms is 5755% of the 28.7 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | jax-cuda | maximum_strided_read_f64_n1048576 untimed 1541.2 ms is 5677% of the 27.2 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | jax-cuda | minimum_strided_read_f64_n1048576 untimed 1580.2 ms is 5663% of the 27.9 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | jax-cuda | abs_strided_read_f64_n1048576 untimed 1579.5 ms is 6421% of the 24.6 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | jax-cuda | sign_strided_read_f64_n1048576 untimed 1559.5 ms is 6528% of the 23.9 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | jax-cuda | expm1_strided_read_f64_n1048576 untimed 1531.4 ms is 6524% of the 23.5 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | jax-cuda | log_strided_read_f64_n1048576 untimed 1525.4 ms is 6431% of the 23.7 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | jax-cuda | log1p_strided_read_f64_n1048576 untimed 1543.8 ms is 6766% of the 22.8 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | jax-cuda | erf_strided_read_f64_n1048576 untimed 1545.7 ms is 4955% of the 31.2 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | jax-cuda | sin_strided_read_f64_n1048576 untimed 1541.7 ms is 6400% of the 24.1 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | jax-cuda | cos_strided_read_f64_n1048576 untimed 1557.9 ms is 5870% of the 26.5 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | jax-cuda | sqrt_strided_read_f64_n1048576 untimed 1558.6 ms is 6512% of the 23.9 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | rsqrt_strided_read_f64_n1048576 untimed 1536.9 ms is 4523% of the 34.0 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | reduce_prod_strided_read_f64_n1048576 untimed 218.8 ms is 417% of the 52.4 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | reduce_max_strided_read_f64_n1048576 untimed 298.4 ms is 578% of the 51.6 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | reduce_min_strided_read_f64_n1048576 untimed 140.3 ms is 242% of the 58.1 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | reduce_sum_squares_strided_read_f64_n1048576 untimed 309.3 ms is 594% of the 52.1 ms timed region (budget 20%) |