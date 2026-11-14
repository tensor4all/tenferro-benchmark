# gpu/reuse_contraction: CUDA public API benchmark

Measured cells: **150**. Unsupported cells: **74**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1024.4 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261114_082852/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**44** of 56) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Transposed borrowed-read cases use prepared native CUDA views in tenferro and PyTorch. JAX compares the same logical values in its native representation; its columns do not claim an equivalent public strided-view contract. View creation and input conversion are outside timing.

Generalized batch dot preserves native output order: tenferro [m,n,b], PyTorch/JAX [b,m,n]. Input permutation views and dimension-number configurations are prepared before timing. Output axes are normalized only on downloaded CPU arrays for verification; no timed output transpose or copy is added.

Prepared einsum uses the public ConcreteEinsumPlan for ik,kj->ij, built once outside execution and sampling. Native matrix multiplication/reuse/accumulation references match the selected mathematical and output-ownership contracts; they do not assert an equivalent prepared-plan interface. Transposed input views/values, read/write handles and accumulation seed restoration are untimed.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| add_into_f64_n1048576 | 38.116 (3.640) | unsupported | 13.645 (4.383) | unsupported |
| add_read_into_f64_n1048576 | 42.839 (13.781) | unsupported | 23.344 (6.744) | unsupported |
| sub_into_f64_n1048576 | 48.383 (18.272) | unsupported | 13.375 (1.554) | unsupported |
| sub_read_into_f64_n1048576 † | 36.492 (17.356) | unsupported | 31.315 (50.600) | unsupported |
| mul_into_f64_n1048576 | 46.174 (18.883) | unsupported | 14.003 (0.938) | unsupported |
| mul_read_into_f64_n1048576 | 39.923 (4.906) | unsupported | 21.345 (11.175) | unsupported |
| div_into_f64_n1048576 | 38.657 (3.933) | unsupported | 14.544 (0.384) | unsupported |
| div_read_into_f64_n1048576 | 57.235 (12.973) | unsupported | 14.641 (0.480) | unsupported |
| neg_into_f64_n1048576 | 31.228 (5.081) | unsupported | 8.948 (0.777) | unsupported |
| neg_read_into_f64_n1048576 † | 52.364 (33.533) | unsupported | 9.516 (3.750) | unsupported |
| conj_into_c64_n1048576 | 56.013 (4.112) | unsupported | 31.760 (12.336) | unsupported |
| conj_read_into_c64_n1048576 | 55.035 (122.676) | unsupported | 27.971 (2.087) | unsupported |
| tensordot_f64_n262144 † | 74.739 (6.491) | 167.165 (77.812) | 25.769 (0.731) | 401.064 (120.564) |
| vdot_read_f64_n262144 † | 251.830 (80.972) | unsupported | 10.170 (0.560) | 473.663 (212.213) |
| norm_squared_read_f64_n262144 † | 66.297 (28.521) | unsupported | 10.130 (0.978) | 347.914 (127.722) |
| axpby_read_into_accum_f64_n1048576 | 52.163 (8.831) | unsupported | 20.839 (6.639) | unsupported |
| dot_general_read_into_f64_s256 † | 35.676 (3.536) | unsupported | 8.523 (1.162) | unsupported |
| dot_general_read_into_accum_f64_s256 † | 67.554 (18.517) | unsupported | 9.557 (0.871) | unsupported |
| dot_general_read_into_accum_cached_f64_s256 † | 43.363 (16.950) | unsupported | 9.612 (0.801) | unsupported |
| add_strided_read_f64_n1048576 † | 187.175 (21.939) | unsupported | 14.396 (1.977) | 1572.825 (523.251) |
| mul_strided_read_f64_n1048576 † | 183.229 (12.283) | unsupported | 13.287 (1.641) | 157.841 (26.889) |
| neg_strided_read_f64_n1048576 † | 118.660 (22.803) | unsupported | 8.983 (1.598) | 164.075 (38.107) |
| exp_strided_read_f64_n1048576 † | 152.106 (31.062) | unsupported | 9.385 (1.763) | 245.108 (92.671) |
| tanh_strided_read_f64_n1048576 † | 114.714 (16.985) | unsupported | 9.324 (0.362) | 2014.502 (2006.959) |
| reduce_sum_strided_read_f64_n1048576 † | 131.511 (23.056) | unsupported | 13.569 (0.272) | 191.191 (49.919) |
| conj_strided_read_c64_n1048576 | 126.098 (13.593) | unsupported | 28.722 (9.752) | 251.638 (55.131) |
| dot_general_batch_f64_s256 † | 61.325 (41.955) | 65.241 (20.184) | 17.153 (0.587) | 194.997 (35.121) |
| dot_conj_l0r0_c64_n262144 † | 68.881 (16.184) | 91.789 (17.888) | 12.910 (0.503) | 399.108 (55.175) |
| einsum_plan_owned_f64_s256 † | 49.509 (4.640) | unsupported | 8.245 (0.405) | 1201.829 (425.054) |
| einsum_plan_read_f64_s256 † | 61.648 (19.874) | unsupported | 8.111 (0.323) | 206.329 (82.339) |
| einsum_plan_strided_read_f64_s256 † | 178.462 (32.511) | unsupported | 8.413 (0.882) | 238.787 (137.378) |
| einsum_plan_into_f64_s256 † | 75.162 (66.613) | unsupported | 11.513 (15.644) | unsupported |
| einsum_plan_read_into_f64_s256 † | 62.779 (25.442) | unsupported | 9.137 (2.443) | unsupported |
| einsum_plan_read_into_accum_f64_s256 † | 37.055 (11.805) | unsupported | 9.991 (1.257) | unsupported |
| einsum_plan_strided_read_into_f64_s256 † | 151.189 (19.023) | unsupported | 7.995 (0.335) | unsupported |
| einsum_plan_strided_read_into_accum_f64_s256 † | 160.216 (35.523) | unsupported | 9.085 (0.361) | unsupported |
| sub_strided_read_f64_n1048576 † | 177.048 (14.300) | unsupported | 13.855 (0.428) | 500.762 (113.387) |
| div_strided_read_f64_n1048576 † | 244.794 (114.854) | unsupported | 14.571 (4.566) | 282.648 (126.308) |
| rem_strided_read_f64_n1048576 † | 332.479 (88.471) | unsupported | 14.125 (1.170) | 283.416 (59.627) |
| pow_strided_read_f64_n1048576 † | 286.105 (62.927) | unsupported | 30.730 (0.860) | 167.305 (55.960) |
| maximum_strided_read_f64_n1048576 † | 216.037 (38.910) | unsupported | 14.073 (0.415) | 548.464 (147.155) |
| minimum_strided_read_f64_n1048576 † | 195.084 (76.790) | unsupported | 23.362 (32.441) | 190.981 (60.343) |
| abs_strided_read_f64_n1048576 † | 127.076 (16.849) | unsupported | 9.376 (2.960) | 373.791 (99.175) |
| sign_strided_read_f64_n1048576 † | 100.947 (6.936) | unsupported | 10.572 (3.213) | 139.089 (38.456) |
| expm1_strided_read_f64_n1048576 † | 104.299 (7.447) | unsupported | 10.983 (0.982) | 423.936 (103.923) |
| log_strided_read_f64_n1048576 † | 139.927 (79.820) | unsupported | 15.330 (4.675) | 145.776 (52.067) |
| log1p_strided_read_f64_n1048576 † | 134.541 (28.382) | unsupported | 14.317 (0.576) | 150.054 (41.023) |
| erf_strided_read_f64_n1048576 † | 378.260 (307.197) | unsupported | 19.870 (18.800) | 553.981 (276.292) |
| sin_strided_read_f64_n1048576 † | 124.492 (21.533) | unsupported | 12.135 (0.663) | 161.316 (39.775) |
| cos_strided_read_f64_n1048576 † | 120.319 (10.878) | unsupported | 12.537 (1.615) | 167.270 (17.914) |
| sqrt_strided_read_f64_n1048576 † | 132.594 (74.355) | unsupported | 9.228 (0.956) | 247.640 (145.279) |
| rsqrt_strided_read_f64_n1048576 † | 145.706 (32.310) | unsupported | 9.691 (2.366) | 459.521 (419.065) |
| reduce_prod_strided_read_f64_n1048576 † | 132.419 (10.467) | unsupported | 13.992 (1.403) | 188.589 (29.866) |
| reduce_max_strided_read_f64_n1048576 † | 126.762 (5.605) | unsupported | 14.303 (1.079) | 334.470 (88.821) |
| reduce_min_strided_read_f64_n1048576 † | 133.310 (14.745) | unsupported | 13.694 (0.241) | 297.803 (57.008) |
| reduce_sum_squares_strided_read_f64_n1048576 † | 124.335 (7.019) | unsupported | 21.572 (0.444) | 3680.636 (1240.661) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_strided_read_f64_n1048576 | jax-cuda | 4 | 15 | 1.495 | 16777216 | 0.044 | 0 |
| abs_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.300 | 16777216 | 1.747 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 4.066 | 16777216 | 0.129 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| add_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.437 | 25165824 | 1.800 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.220 | 25165824 | 0.645 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.747 | 25165824 | 1.052 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.371 | 25165824 | 0.573 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_strided_read_f64_n1048576 | jax-cuda | 1 | 15 | 1.573 | 25165824 | 0.016 | 0 |
| add_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.461 | 25165824 | 1.706 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.995 | 25165824 | 0.131 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| axpby_read_into_accum_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.667 | 25165824 | 1.179 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.669 | 25165824 | 0.471 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No matching public caller-owned accumulating output API; donation is not output reuse. |
| conj_into_c64_n1048576 | jax-cuda | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.508 | 33554432 | 1.031 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.896 | 33554432 | 0.585 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1048576 | jax-cuda | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.448 | 33554432 | 1.171 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.881 | 33554432 | 0.595 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_strided_read_c64_n1048576 | jax-cuda | 16 | 15 | 4.026 | 33554432 | 0.130 | 0 |
| conj_strided_read_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.460 | 33554432 | 1.140 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.018 | 33554432 | 0.260 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| cos_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.676 | 16777216 | 0.098 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.401 | 16777216 | 1.306 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.925 | 16777216 | 0.136 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| div_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.465 | 25165824 | 1.689 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.237 | 25165824 | 0.636 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.468 | 25165824 | 1.678 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.832 | 25165824 | 0.429 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_strided_read_f64_n1048576 | jax-cuda | 4 | 15 | 1.131 | 25165824 | 0.087 | 0 |
| div_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.466 | 25165824 | 1.686 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.917 | 25165824 | 0.100 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| dot_conj_l0r0_c64_n262144 | jax-cuda | 8 | 15 | 3.193 | 8388624 | 0.021 | 1.221e-05 |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | 256 | 15 | 3.305 | 8388624 | 0.634 | 1.366e-05 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | 64 | 15 | 4.408 | 8388624 | 0.119 | 1.212e-05 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | 16 | 15 | 1.469 | 8388624 | 0.089 | 1.212e-05 |
| dot_general_batch_f64_s256 | jax-cuda | 16 | 15 | 3.120 | 3210240 | 0.016 | 0 |
| dot_general_batch_f64_s256 | pytorch-cuda | 128 | 15 | 2.196 | 3210240 | 0.183 | 0 |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | 64 | 15 | 3.925 | 3210240 | 0.051 | 0 |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | 32 | 15 | 2.088 | 3210240 | 0.048 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | 256 | 15 | 2.461 | 2097152 | 0.213 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | 64 | 15 | 2.775 | 2097152 | 0.047 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.447 | 2097152 | 0.214 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | 32 | 15 | 2.162 | 2097152 | 0.030 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.182 | 2097152 | 0.240 | 0 |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | 64 | 15 | 2.283 | 2097152 | 0.057 | 0 |
| dot_general_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| einsum_plan_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.947 | 1070080 | 0.091 | 1.093e-06 |
| einsum_plan_into_f64_s256 | tenferro-cuda-direct | 64 | 15 | 4.810 | 1070080 | 0.014 | 1.093e-06 |
| einsum_plan_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_owned_f64_s256 | jax-cuda | 1 | 15 | 1.202 | 1070080 | 0.001 | 1.093e-06 |
| einsum_plan_owned_f64_s256 | pytorch-cuda | 256 | 15 | 2.111 | 1070080 | 0.127 | 1.093e-06 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-direct | 64 | 15 | 3.169 | 1070080 | 0.021 | 1.093e-06 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_f64_s256 | jax-cuda | 32 | 15 | 6.603 | 1070080 | 0.005 | 1.093e-06 |
| einsum_plan_read_f64_s256 | pytorch-cuda | 256 | 15 | 2.077 | 1070080 | 0.129 | 1.093e-06 |
| einsum_plan_read_f64_s256 | tenferro-cuda-direct | 64 | 15 | 3.945 | 1070080 | 0.017 | 1.093e-06 |
| einsum_plan_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_accum_f64_s256 | jax-cuda | — | — | — | 1608704 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.558 | 1608704 | 0.157 | 3.261e-07 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-direct | 64 | 15 | 2.372 | 1608704 | 0.042 | 3.261e-07 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.339 | 1070080 | 0.114 | 1.093e-06 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-direct | 64 | 15 | 4.018 | 1070080 | 0.017 | 1.093e-06 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_f64_s256 | jax-cuda | 4 | 15 | 0.955 | 1070080 | 0.004 | 1.093e-06 |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | 256 | 15 | 2.154 | 1070080 | 0.124 | 1.093e-06 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.855 | 1070080 | 0.006 | 1.093e-06 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_accum_f64_s256 | jax-cuda | — | — | — | 1608704 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.326 | 1608704 | 0.173 | 3.261e-07 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.563 | 1608704 | 0.010 | 3.261e-07 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.047 | 1070080 | 0.131 | 1.093e-06 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.419 | 1070080 | 0.007 | 1.093e-06 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| erf_strided_read_f64_n1048576 | jax-cuda | 2 | 15 | 1.108 | 16777216 | 0.030 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.636 | 16777216 | 0.824 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 1.513 | 16777216 | 0.043 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| exp_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 3.922 | 16777216 | 0.067 | 1.093e-07 |
| exp_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.300 | 16777216 | 1.745 | 1.093e-07 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 4.867 | 16777216 | 0.108 | 1.093e-07 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| expm1_strided_read_f64_n1048576 | jax-cuda | 2 | 15 | 0.848 | 16777216 | 0.039 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.351 | 16777216 | 1.491 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.338 | 16777216 | 0.157 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log1p_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.401 | 16777216 | 0.109 | 1.106e-07 |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.458 | 16777216 | 1.144 | 1.106e-07 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 4.305 | 16777216 | 0.122 | 1.106e-07 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.332 | 16777216 | 0.112 | 4.308e-08 |
| log_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.491 | 16777216 | 1.068 | 4.308e-08 |
| log_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.239 | 16777216 | 0.117 | 4.308e-08 |
| log_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| maximum_strided_read_f64_n1048576 | jax-cuda | 4 | 15 | 2.194 | 25165824 | 0.045 | 0 |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.450 | 25165824 | 1.746 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.457 | 25165824 | 0.114 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| minimum_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 3.056 | 25165824 | 0.129 | 0 |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.748 | 25165824 | 1.052 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.121 | 25165824 | 0.126 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| mul_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.448 | 25165824 | 1.754 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.478 | 25165824 | 0.532 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.683 | 25165824 | 1.151 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.278 | 25165824 | 0.615 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.525 | 25165824 | 0.156 | 0 |
| mul_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.425 | 25165824 | 1.849 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.932 | 25165824 | 0.134 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| neg_into_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.286 | 16777216 | 1.830 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.999 | 16777216 | 0.524 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.305 | 16777216 | 1.721 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.838 | 16777216 | 0.313 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.625 | 16777216 | 0.100 | 0 |
| neg_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.287 | 16777216 | 1.823 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.899 | 16777216 | 0.138 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| norm_squared_read_f64_n262144 | jax-cuda | 8 | 15 | 2.783 | 2097160 | 0.006 | 4.991e-07 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.593 | 2097160 | 0.202 | 4.991e-07 |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | 32 | 15 | 2.121 | 2097160 | 0.031 | 4.991e-07 |
| norm_squared_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 2097160 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| pow_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.677 | 25165824 | 0.147 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.983 | 25165824 | 0.799 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.289 | 25165824 | 0.086 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | 8 | 15 | 2.676 | 8392704 | 0.024 | 0 |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.662 | 8392704 | 0.573 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.028 | 8392704 | 0.065 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | 8 | 15 | 2.382 | 8392704 | 0.028 | 0 |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.506 | 8392704 | 0.598 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.133 | 8392704 | 0.061 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 3.017 | 8392704 | 0.043 | 4.995e-06 |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.582 | 8392704 | 0.586 | 4.106e-06 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.119 | 8392704 | 0.062 | 4.661e-06 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | 1 | 15 | 3.681 | 8392704 | 0.002 | 3.311e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.761 | 8392704 | 0.380 | 3.309e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.989 | 8392704 | 0.066 | 3.312e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | 8 | 15 | 1.530 | 8392704 | 0.043 | 7.159e-06 |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.474 | 8392704 | 0.604 | 5.935e-06 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.104 | 8392704 | 0.062 | 4.724e-06 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rem_strided_read_f64_n1048576 | jax-cuda | 8 | 15 | 2.267 | 25165824 | 0.087 | 0 |
| rem_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.452 | 25165824 | 1.739 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.660 | 25165824 | 0.074 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | 8 | 15 | 3.676 | 16777216 | 0.036 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.310 | 16777216 | 1.690 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.331 | 16777216 | 0.112 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sign_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.225 | 16777216 | 0.118 | 0 |
| sign_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.338 | 16777216 | 1.549 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.230 | 16777216 | 0.162 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sin_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.581 | 16777216 | 0.102 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.388 | 16777216 | 1.350 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.984 | 16777216 | 0.132 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sqrt_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 3.962 | 16777216 | 0.066 | 0 |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.295 | 16777216 | 1.775 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.121 | 16777216 | 0.124 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sub_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.428 | 25165824 | 1.837 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.548 | 25165824 | 0.508 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.002 | 25165824 | 0.784 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 0.292 | 25165824 | 0.673 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_strided_read_f64_n1048576 | jax-cuda | 8 | 15 | 4.006 | 25165824 | 0.049 | 0 |
| sub_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.443 | 25165824 | 1.773 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.833 | 25165824 | 0.139 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tanh_strided_read_f64_n1048576 | jax-cuda | 1 | 15 | 2.015 | 16777216 | 0.008 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.298 | 16777216 | 1.757 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.671 | 16777216 | 0.143 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tensordot_f64_n262144 | jax-cuda | 8 | 15 | 3.209 | 6291456 | 0.015 | 6.714e-06 |
| tensordot_f64_n262144 | pytorch-cuda | 128 | 15 | 3.298 | 6291456 | 0.238 | 6.714e-06 |
| tensordot_f64_n262144 | tenferro-cuda-direct | 32 | 15 | 2.392 | 6291456 | 0.082 | 6.714e-06 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 16 | 15 | 2.675 | 6291456 | 0.037 | 6.714e-06 |
| vdot_read_f64_n262144 | jax-cuda | 4 | 15 | 1.895 | 4194312 | 0.009 | 2.995e-06 |
| vdot_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.603 | 4194312 | 0.403 | 2.995e-06 |
| vdot_read_f64_n262144 | tenferro-cuda-direct | 16 | 15 | 4.029 | 4194312 | 0.016 | 2.995e-06 |
| vdot_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 4194312 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261114_082852/gpu_reuse_contraction.yaml
timestamp: '2026-11-14T08:28:52+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: 76cb137693f0b16de2556cc47d32686deaa05863
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
  cpu: AMD EPYC-Genoa Processor
  logical_cpus: 116
  power:
    smt:
      active: '1'
      control: 'on'
  numa:
    nodes: 1
    cpulists:
      '0': 0-115
  smt_siblings:
  - 0-1
  - 2-3
  - 4-5
  - 6-7
  - 8-9
  - 10-11
  - 12-13
  - 14-15
  - 16-17
  - 18-19
  - 20-21
  - 22-23
  - 24-25
  - 26-27
  - 28-29
  - 30-31
  - 32-33
  - 34-35
  - 36-37
  - 38-39
  - 40-41
  - 42-43
  - 44-45
  - 46-47
  - 48-49
  - 50-51
  - 52-53
  - 54-55
  - 56-57
  - 58-59
  - 60-61
  - 62-63
  - 64-65
  - 66-67
  - 68-69
  - 70-71
  - 72-73
  - 74-75
  - 76-77
  - 78-79
  - 80-81
  - 82-83
  - 84-85
  - 86-87
  - 88-89
  - 90-91
  - 92-93
  - 94-95
  - 96-97
  - 98-99
  - 100-101
  - 102-103
  - 104-105
  - 106-107
  - 108-109
  - 110-111
  - 112-113
  - 114-115
  l3_domains:
  - 0-115
build:
  toolchain:
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: dba65816ff33
  os: Linux-6.8.0-147-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-115
    l3_domains:
    - 0-115
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
    GPU_RUNTIME_SHA256: f876805d95374305db5787a72cee40a6848d0fc64ee71d7b306c264971c36869
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
  device_uuid: GPU-c70851ec-1874-5bb5-8a0a-85e4eed17212
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
  - 0.130184
  - 0.200026
  - 0.147225
  - 0.131022
  - 0.129346
  - 0.121803
  - 0.254781
  - 0.132419
  - 0.153371
  - 0.138286
  - 0.120965
  - 0.14946
  - 0.125435
  - 0.121524
  - 0.130464
  median_ms: 0.131022
  bytes_per_second: 1024390774068.477
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-14T08:49:13.393923+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-c70851ec-1874-5bb5-8a0a-85e4eed17212
  cuda_visible_devices: GPU-c70851ec-1874-5bb5-8a0a-85e4eed17212
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
    src/bin/benchmark_gpu_public_api.rs: 3c723c1d0c1658d8499a7713ced3b30357447f661dae37fca7fa4ce2c72de804
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 9c3d7916a3281d2dfd1caa68551dc9c81c79dab6b7525324339e5c4816581e6d
    src/bin/benchmark_gpu_public_api/multi.rs: 512b659c731722a152dbdf2e41569974aa84442dfc1f739afbe234e08c71c93b
    src/bin/benchmark_gpu_public_api/plan.rs: 70fce8a6e9ed7518779a51a9d8692e010fa1e470e95eee0ce5276f595737fde8
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: a5a4aa0edb0a1158f6669b642f4bc25a7079314fc2a711ef3ef57cc19c0a35b2
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: c9d61b1e99bda517a59a6e34f625fd4d651bbbb19f6dc06ba4d0d1c0d50dfd6a
    src/bin/benchmark_gpu_public_api/state.rs: a8bbf8985bbc58c435e3c4dda7135161a141a90a10fa50d4f95128397ce949cb
    src/bin/benchmark_gpu_public_api/structural.rs: 8b61ac01b58edc090eafbc124ec135d5efeb4979631a939656ba78dd6e5c042a
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: d1ff68f86496d991bcbc730138fe6e3076001b3f23cc1a94c44b96f6857cb8ed
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: b0e1cdcefc95a1f2170739eeffbc2e9600eb3acd363566b605458401b04ec37e
    data/results/nvidia-gpu/gpu/reuse_contraction/20261114_082852/fixtures/manifest.json: 63a15555590136dc9184229fe03dd2eea2bd9e21f80607686d700c24c41fde1f
    target/release/benchmark_gpu_public_api: 25f583eab659e7cc2c88a104c8203f0c2f400fd4316bfed75f479ee99d3f16bb
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-14T08:50:11Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-14T08:49:52Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-14T08:49:36Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-14T08:49:42Z'
  compile:
    case_wall_seconds: 44.682
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
| add_into_f64_n1048576 | tenferro-cuda-direct | add_into_f64_n1048576 untimed 384.5 ms is 2118% of the 18.2 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | add_read_into_f64_n1048576 untimed 474.2 ms is 880% of the 53.9 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | tenferro-cuda-direct | sub_into_f64_n1048576 untimed 430.9 ms is 1745% of the 24.7 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | sub_read_into_f64_n1048576 untimed 369.8 ms is 1570% of the 23.6 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | tenferro-cuda-direct | mul_into_f64_n1048576 untimed 434.6 ms is 1010% of the 43.0 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | mul_read_into_f64_n1048576 untimed 352.1 ms is 1801% of the 19.6 ms timed region (budget 20%) |
| div_into_f64_n1048576 | tenferro-cuda-direct | div_into_f64_n1048576 untimed 289.4 ms is 1537% of the 18.8 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | div_read_into_f64_n1048576 untimed 371.5 ms is 1288% of the 28.8 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | tenferro-cuda-direct | neg_into_f64_n1048576 untimed 279.7 ms is 1785% of the 15.7 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | neg_read_into_f64_n1048576 untimed 331.3 ms is 2054% of the 16.1 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | tenferro-cuda-direct | conj_into_c64_n1048576 untimed 329.9 ms is 2443% of the 13.5 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | conj_read_into_c64_n1048576 untimed 400.8 ms is 1567% of the 25.6 ms timed region (budget 20%) |
| tensordot_f64_n262144 | tenferro-cuda-direct | tensordot_f64_n262144 untimed 3144.6 ms is 8599% of the 36.6 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | tenferro-cuda-direct | vdot_read_f64_n262144 untimed 25.3 ms is 47% of the 54.2 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | norm_squared_read_f64_n262144 untimed 7.4 ms is 25% of the 30.1 ms timed region (budget 20%) |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | axpby_read_into_accum_f64_n1048576 untimed 351.8 ms is 1374% of the 25.6 ms timed region (budget 20%) |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | dot_general_read_into_f64_s256 untimed 250.5 ms is 710% of the 35.3 ms timed region (budget 20%) |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | dot_general_read_into_accum_f64_s256 untimed 435.8 ms is 1235% of the 35.3 ms timed region (budget 20%) |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | dot_general_read_into_accum_cached_f64_s256 untimed 459.2 ms is 721% of the 63.7 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | add_strided_read_f64_n1048576 untimed 139.0 ms is 266% of the 52.2 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | mul_strided_read_f64_n1048576 untimed 159.4 ms is 364% of the 43.7 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | neg_strided_read_f64_n1048576 untimed 93.2 ms is 193% of the 48.3 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | exp_strided_read_f64_n1048576 untimed 99.1 ms is 141% of the 70.1 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | tanh_strided_read_f64_n1048576 untimed 88.7 ms is 161% of the 55.2 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_sum_strided_read_f64_n1048576 untimed 17.3 ms is 53% of the 32.5 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | conj_strided_read_c64_n1048576 untimed 135.9 ms is 417% of the 32.6 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | dot_general_batch_f64_s256 untimed 37.7 ms is 41% of the 92.3 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | dot_conj_l0r0_c64_n262144 untimed 21.5 ms is 29% of the 73.1 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_f64_s256 untimed 9.1 ms is 22% of the 41.8 ms timed region (budget 20%) |
| einsum_plan_into_f64_s256 | tenferro-cuda-direct | einsum_plan_into_f64_s256 untimed 550.5 ms is 413% of the 133.2 ms timed region (budget 20%) |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-direct | einsum_plan_read_into_f64_s256 untimed 756.1 ms is 1125% of the 67.2 ms timed region (budget 20%) |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-direct | einsum_plan_read_into_accum_f64_s256 untimed 860.5 ms is 1359% of the 63.3 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_into_f64_s256 untimed 632.5 ms is 1726% of the 36.6 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_into_accum_f64_s256 untimed 636.8 ms is 1116% of the 57.1 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | tenferro-cuda-direct | sub_strided_read_f64_n1048576 untimed 146.9 ms is 265% of the 55.4 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | tenferro-cuda-direct | div_strided_read_f64_n1048576 untimed 154.3 ms is 184% of the 83.7 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | tenferro-cuda-direct | rem_strided_read_f64_n1048576 untimed 154.6 ms is 243% of the 63.6 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | tenferro-cuda-direct | pow_strided_read_f64_n1048576 untimed 152.9 ms is 450% of the 34.0 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-direct | maximum_strided_read_f64_n1048576 untimed 136.2 ms is 227% of the 59.9 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-direct | minimum_strided_read_f64_n1048576 untimed 122.8 ms is 234% of the 52.6 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | tenferro-cuda-direct | abs_strided_read_f64_n1048576 untimed 103.6 ms is 168% of the 61.8 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | tenferro-cuda-direct | sign_strided_read_f64_n1048576 untimed 95.8 ms is 194% of the 49.5 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-direct | expm1_strided_read_f64_n1048576 untimed 94.8 ms is 188% of the 50.5 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | tenferro-cuda-direct | log_strided_read_f64_n1048576 untimed 147.2 ms is 240% of the 61.3 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-direct | log1p_strided_read_f64_n1048576 untimed 157.6 ms is 177% of the 89.1 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | tenferro-cuda-direct | erf_strided_read_f64_n1048576 untimed 142.1 ms is 498% of the 28.5 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | tenferro-cuda-direct | sin_strided_read_f64_n1048576 untimed 112.5 ms is 123% of the 91.1 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | tenferro-cuda-direct | cos_strided_read_f64_n1048576 untimed 131.6 ms is 463% of the 28.4 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | sqrt_strided_read_f64_n1048576 untimed 131.9 ms is 271% of the 48.6 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | rsqrt_strided_read_f64_n1048576 untimed 105.6 ms is 319% of the 33.1 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_prod_strided_read_f64_n1048576 untimed 17.6 ms is 54% of the 32.5 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_max_strided_read_f64_n1048576 untimed 19.7 ms is 62% of the 31.6 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_min_strided_read_f64_n1048576 untimed 16.3 ms is 49% of the 33.0 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_sum_squares_strided_read_f64_n1048576 untimed 17.8 ms is 59% of the 30.3 ms timed region (budget 20%) |
| tensordot_f64_n262144 | tenferro-cuda-eager | tensordot_f64_n262144 untimed 4352.9 ms is 10394% of the 41.9 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | dot_general_batch_f64_s256 untimed 38.5 ms is 97% of the 39.6 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | dot_conj_l0r0_c64_n262144 untimed 24.7 ms is 108% of the 22.9 ms timed region (budget 20%) |
| add_into_f64_n1048576 | pytorch-cuda | add_into_f64_n1048576 untimed 234.3 ms is 2978% of the 7.9 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | pytorch-cuda | add_read_into_f64_n1048576 untimed 135.3 ms is 1224% of the 11.1 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | pytorch-cuda | sub_into_f64_n1048576 untimed 82.7 ms is 1217% of the 6.8 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | pytorch-cuda | sub_read_into_f64_n1048576 untimed 74.8 ms is 252% of the 29.7 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | pytorch-cuda | mul_into_f64_n1048576 untimed 94.9 ms is 1380% of the 6.9 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | pytorch-cuda | mul_read_into_f64_n1048576 untimed 111.7 ms is 1052% of the 10.6 ms timed region (budget 20%) |
| div_into_f64_n1048576 | pytorch-cuda | div_into_f64_n1048576 untimed 95.2 ms is 1364% of the 7.0 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | pytorch-cuda | div_read_into_f64_n1048576 untimed 92.1 ms is 1259% of the 7.3 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | pytorch-cuda | neg_into_f64_n1048576 untimed 82.4 ms is 1702% of the 4.8 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | pytorch-cuda | neg_read_into_f64_n1048576 untimed 91.3 ms is 1691% of the 5.4 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | pytorch-cuda | conj_into_c64_n1048576 untimed 98.9 ms is 1225% of the 8.1 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | pytorch-cuda | conj_read_into_c64_n1048576 untimed 79.9 ms is 1165% of the 6.9 ms timed region (budget 20%) |
| tensordot_f64_n262144 | pytorch-cuda | tensordot_f64_n262144 untimed 78.1 ms is 154% of the 50.7 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | pytorch-cuda | vdot_read_f64_n262144 untimed 13.4 ms is 34% of the 39.3 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | pytorch-cuda | norm_squared_read_f64_n262144 untimed 11.0 ms is 27% of the 40.3 ms timed region (budget 20%) |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | axpby_read_into_accum_f64_n1048576 untimed 81.1 ms is 744% of the 10.9 ms timed region (budget 20%) |
| dot_general_read_into_f64_s256 | pytorch-cuda | dot_general_read_into_f64_s256 untimed 39.0 ms is 108% of the 36.0 ms timed region (budget 20%) |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | dot_general_read_into_accum_f64_s256 untimed 263.0 ms is 707% of the 37.2 ms timed region (budget 20%) |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | dot_general_read_into_accum_cached_f64_s256 untimed 42.8 ms is 111% of the 38.7 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | pytorch-cuda | add_strided_read_f64_n1048576 untimed 83.6 ms is 1158% of the 7.2 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | pytorch-cuda | mul_strided_read_f64_n1048576 untimed 75.3 ms is 1118% of the 6.7 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | pytorch-cuda | neg_strided_read_f64_n1048576 untimed 61.9 ms is 1314% of the 4.7 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | pytorch-cuda | exp_strided_read_f64_n1048576 untimed 77.6 ms is 1578% of the 4.9 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | tanh_strided_read_f64_n1048576 untimed 64.3 ms is 1423% of the 4.5 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_strided_read_f64_n1048576 untimed 27.0 ms is 51% of the 52.8 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | pytorch-cuda | conj_strided_read_c64_n1048576 untimed 123.8 ms is 1581% of the 7.8 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | pytorch-cuda | dot_general_batch_f64_s256 untimed 31.5 ms is 94% of the 33.5 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | dot_conj_l0r0_c64_n262144 untimed 13.6 ms is 27% of the 50.3 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | pytorch-cuda | einsum_plan_owned_f64_s256 untimed 14.7 ms is 45% of the 32.9 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | pytorch-cuda | einsum_plan_read_f64_s256 untimed 13.4 ms is 42% of the 31.4 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | einsum_plan_strided_read_f64_s256 untimed 13.6 ms is 39% of the 34.4 ms timed region (budget 20%) |
| einsum_plan_into_f64_s256 | pytorch-cuda | einsum_plan_into_f64_s256 untimed 18.6 ms is 27% of the 68.1 ms timed region (budget 20%) |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | einsum_plan_read_into_f64_s256 untimed 177.1 ms is 467% of the 37.9 ms timed region (budget 20%) |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_read_into_accum_f64_s256 untimed 36.4 ms is 94% of the 38.6 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_f64_s256 untimed 13.8 ms is 45% of the 31.0 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_accum_f64_s256 untimed 35.0 ms is 99% of the 35.4 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | pytorch-cuda | sub_strided_read_f64_n1048576 untimed 68.7 ms is 1035% of the 6.6 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | pytorch-cuda | div_strided_read_f64_n1048576 untimed 95.1 ms is 1166% of the 8.1 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | pytorch-cuda | rem_strided_read_f64_n1048576 untimed 95.0 ms is 1365% of the 7.0 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | pytorch-cuda | pow_strided_read_f64_n1048576 untimed 96.5 ms is 637% of the 15.1 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | maximum_strided_read_f64_n1048576 untimed 118.4 ms is 1742% of the 6.8 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | minimum_strided_read_f64_n1048576 untimed 66.3 ms is 334% of the 19.8 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | pytorch-cuda | abs_strided_read_f64_n1048576 untimed 58.1 ms is 1159% of the 5.0 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | pytorch-cuda | sign_strided_read_f64_n1048576 untimed 68.6 ms is 1211% of the 5.7 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | expm1_strided_read_f64_n1048576 untimed 82.4 ms is 1480% of the 5.6 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | pytorch-cuda | log_strided_read_f64_n1048576 untimed 61.9 ms is 752% of the 8.2 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | log1p_strided_read_f64_n1048576 untimed 64.2 ms is 880% of the 7.3 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | pytorch-cuda | erf_strided_read_f64_n1048576 untimed 86.3 ms is 636% of the 13.6 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | pytorch-cuda | sin_strided_read_f64_n1048576 untimed 65.6 ms is 1090% of the 6.0 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | pytorch-cuda | cos_strided_read_f64_n1048576 untimed 62.7 ms is 970% of the 6.5 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | sqrt_strided_read_f64_n1048576 untimed 62.5 ms is 1367% of the 4.6 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | rsqrt_strided_read_f64_n1048576 untimed 63.5 ms is 1185% of the 5.4 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | reduce_prod_strided_read_f64_n1048576 untimed 787.4 ms is 1434% of the 54.9 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | reduce_max_strided_read_f64_n1048576 untimed 23.1 ms is 42% of the 55.0 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | reduce_min_strided_read_f64_n1048576 untimed 24.1 ms is 46% of the 52.9 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_squares_strided_read_f64_n1048576 untimed 19.6 ms is 47% of the 41.6 ms timed region (budget 20%) |
| tensordot_f64_n262144 | jax-cuda | tensordot_f64_n262144 untimed 1259.6 ms is 1810% of the 69.6 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | jax-cuda | vdot_read_f64_n262144 untimed 490.1 ms is 1732% of the 28.3 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | jax-cuda | norm_squared_read_f64_n262144 untimed 25.9 ms is 61% of the 42.2 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | jax-cuda | add_strided_read_f64_n1048576 untimed 714.2 ms is 3195% of the 22.4 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | jax-cuda | mul_strided_read_f64_n1048576 untimed 314.0 ms is 811% of the 38.7 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | jax-cuda | neg_strided_read_f64_n1048576 untimed 293.1 ms is 698% of the 42.0 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | jax-cuda | exp_strided_read_f64_n1048576 untimed 360.3 ms is 579% of the 62.3 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | jax-cuda | tanh_strided_read_f64_n1048576 untimed 398.6 ms is 917% of the 43.5 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | reduce_sum_strided_read_f64_n1048576 untimed 407.7 ms is 1645% of the 24.8 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | jax-cuda | conj_strided_read_c64_n1048576 untimed 566.1 ms is 898% of the 63.0 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | jax-cuda | dot_general_batch_f64_s256 untimed 991.6 ms is 1327% of the 74.7 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | jax-cuda | dot_conj_l0r0_c64_n262144 untimed 212.4 ms is 447% of the 47.6 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | jax-cuda | einsum_plan_owned_f64_s256 untimed 785.1 ms is 3931% of the 20.0 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | jax-cuda | einsum_plan_read_f64_s256 untimed 52.0 ms is 37% of the 142.2 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | jax-cuda | einsum_plan_strided_read_f64_s256 untimed 742.9 ms is 2358% of the 31.5 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | jax-cuda | sub_strided_read_f64_n1048576 untimed 404.1 ms is 543% of the 74.4 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | jax-cuda | div_strided_read_f64_n1048576 untimed 356.3 ms is 1739% of the 20.5 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | jax-cuda | rem_strided_read_f64_n1048576 untimed 462.4 ms is 1381% of the 33.5 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | jax-cuda | pow_strided_read_f64_n1048576 untimed 385.3 ms is 661% of the 58.3 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | jax-cuda | maximum_strided_read_f64_n1048576 untimed 325.3 ms is 975% of the 33.4 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | jax-cuda | minimum_strided_read_f64_n1048576 untimed 286.2 ms is 572% of the 50.0 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | jax-cuda | abs_strided_read_f64_n1048576 untimed 290.2 ms is 1190% of the 24.4 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | jax-cuda | sign_strided_read_f64_n1048576 untimed 270.2 ms is 457% of the 59.2 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | jax-cuda | expm1_strided_read_f64_n1048576 untimed 370.7 ms is 3152% of the 11.8 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | jax-cuda | log_strided_read_f64_n1048576 untimed 311.0 ms is 844% of the 36.8 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | jax-cuda | log1p_strided_read_f64_n1048576 untimed 358.4 ms is 954% of the 37.5 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | jax-cuda | erf_strided_read_f64_n1048576 untimed 308.3 ms is 1433% of the 21.5 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | jax-cuda | sin_strided_read_f64_n1048576 untimed 343.9 ms is 848% of the 40.6 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | jax-cuda | cos_strided_read_f64_n1048576 untimed 288.0 ms is 708% of the 40.7 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | jax-cuda | sqrt_strided_read_f64_n1048576 untimed 332.8 ms is 534% of the 62.3 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | rsqrt_strided_read_f64_n1048576 untimed 314.1 ms is 502% of the 62.6 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | reduce_prod_strided_read_f64_n1048576 untimed 373.3 ms is 723% of the 51.6 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | reduce_max_strided_read_f64_n1048576 untimed 408.5 ms is 993% of the 41.1 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | reduce_min_strided_read_f64_n1048576 untimed 142.7 ms is 387% of the 36.9 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | reduce_sum_squares_strided_read_f64_n1048576 untimed 502.4 ms is 680% of the 73.9 ms timed region (budget 20%) |