# gpu/reuse_contraction: CUDA public API benchmark

Measured cells: **189**. Unsupported cells: **35**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1445.1 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261126_062950/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**19** of 56) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Transposed-read cases use prepared native CUDA views in tenferro and PyTorch. JAX compares the same logical values in its native representation; its columns do not claim an equivalent public strided-view contract. View creation and input conversion are outside timing.

- `tenferro-cuda-direct` transposed reads: direct: Engine::*_read on borrowed TensorRead (read_layout::run); transposed borrowed TensorRead over prepared CUDA storage; view and read-handle construction outside timing.
- `tenferro-cuda-eager` transposed reads: eager: EagerSession op on transposed EagerTensor views (read_layout::run_eager); no TensorRead is built; EagerTensor values transposed once by EagerSession::transpose outside timing; that call copies each value into a fresh compact device buffer (one untimed D2D copy per input, which the direct arm does not make) and returns a permuted view of the copy; the timed region is only the eager operation on those permuted views.
- `pytorch-cuda` transposed reads: prepared transposed CUDA view; no input copy inside timing.
- `jax-cuda` transposed reads: same logical values prepared in native JAX representation; no public strided-view equivalence claimed.

Generalized batch dot preserves native output order: tenferro [m,n,b], PyTorch/JAX [b,m,n]. Input permutation views and dimension-number configurations are prepared before timing. Output axes are normalized only on downloaded CPU arrays for verification; no timed output transpose or copy is added.

Prepared einsum uses the public ConcreteEinsumPlan for ik,kj->ij, built once outside execution and sampling. Native matrix multiplication/reuse/accumulation references match the selected mathematical and output-ownership contracts; they do not assert an equivalent prepared-plan interface. Transposed input views/values, read/write handles and accumulation seed restoration are untimed.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| add_into_f64_n1048576 | 12.978 (0.277) | 12.845 (0.138) | 11.690 (0.088) | unsupported |
| add_read_into_f64_n1048576 | 12.868 (0.204) | 12.782 (0.521) | 11.668 (0.225) | unsupported |
| sub_into_f64_n1048576 | 12.942 (0.255) | 12.818 (0.214) | 11.517 (0.301) | unsupported |
| sub_read_into_f64_n1048576 | 12.822 (0.256) | 12.860 (0.140) | 11.698 (0.247) | unsupported |
| mul_into_f64_n1048576 | 12.845 (0.341) | 12.841 (0.222) | 11.440 (0.364) | unsupported |
| mul_read_into_f64_n1048576 | 13.042 (0.309) | 12.850 (0.220) | 11.976 (0.228) | unsupported |
| div_into_f64_n1048576 | 12.909 (0.279) | 12.716 (0.110) | 13.064 (0.165) | unsupported |
| div_read_into_f64_n1048576 | 13.327 (0.265) | 12.658 (0.247) | 13.271 (0.087) | unsupported |
| neg_into_f64_n1048576 | 8.650 (0.051) | 8.641 (0.025) | 7.532 (0.050) | unsupported |
| neg_read_into_f64_n1048576 | 8.653 (0.024) | 8.695 (0.054) | 7.589 (0.064) | unsupported |
| conj_into_c64_n1048576 | 23.837 (0.063) | 23.784 (0.088) | 23.875 (0.185) | unsupported |
| conj_read_into_c64_n1048576 | 23.752 (0.102) | 23.775 (0.083) | 23.916 (0.272) | unsupported |
| tensordot_f64_n262144 † | 28.500 (0.895) | 28.841 (0.442) | 26.027 (0.911) | 128.584 (8.515) |
| vdot_read_f64_n262144 † | 11.269 (0.089) | unsupported | 9.725 (0.106) | 60.328 (0.699) |
| norm_squared_read_f64_n262144 † | 10.933 (0.045) | unsupported | 9.280 (0.013) | 60.468 (1.236) |
| axpby_read_into_accum_f64_n1048576 † | 140.307 (2.789) | unsupported | 15.357 (0.085) | unsupported |
| dot_general_read_into_f64_s256 † | 7.511 (0.025) | unsupported | 9.487 (0.236) | unsupported |
| dot_general_read_into_accum_f64_s256 | 8.787 (0.019) | unsupported | 10.320 (0.176) | unsupported |
| dot_general_read_into_accum_cached_f64_s256 | 8.792 (0.008) | unsupported | 10.443 (0.305) | unsupported |
| add_strided_read_f64_n1048576 | 34.511 (0.067) | 34.304 (0.074) | 12.170 (0.119) | 63.475 (1.022) |
| mul_strided_read_f64_n1048576 | 34.548 (0.053) | 34.039 (0.050) | 11.833 (0.531) | 63.409 (1.846) |
| neg_strided_read_f64_n1048576 | 18.966 (0.036) | 19.019 (0.025) | 7.556 (0.050) | 60.443 (0.678) |
| exp_strided_read_f64_n1048576 | 19.012 (0.033) | 19.096 (0.029) | 8.200 (0.066) | 51.968 (1.270) |
| tanh_strided_read_f64_n1048576 | 19.038 (0.027) | 19.106 (0.042) | 8.101 (0.047) | 49.981 (2.012) |
| reduce_sum_strided_read_f64_n1048576 † | 35.920 (0.089) | 35.647 (0.052) | 13.640 (0.110) | 83.237 (1.309) |
| conj_strided_read_c64_n1048576 | 29.623 (0.114) | 29.386 (0.183) | 23.927 (0.202) | 49.555 (0.833) |
| dot_general_batch_f64_s256 † | 12.641 (0.109) | 14.781 (0.168) | 18.978 (0.299) | 88.186 (2.897) |
| dot_conj_l0r0_c64_n262144 † | 24.281 (0.014) | 24.220 (0.011) | 12.572 (0.010) | 57.243 (2.113) |
| einsum_plan_owned_f64_s256 † | 13.209 (0.067) | unsupported | 10.554 (0.344) | 56.242 (2.100) |
| einsum_plan_read_f64_s256 † | 12.649 (0.070) | unsupported | 10.227 (0.176) | 54.891 (1.760) |
| einsum_plan_strided_read_f64_s256 † | 33.064 (0.380) | unsupported | 9.970 (0.068) | 54.760 (2.269) |
| einsum_plan_into_f64_s256 † | 7.099 (0.408) | unsupported | 9.440 (0.727) | unsupported |
| einsum_plan_read_into_f64_s256 † | 7.415 (0.074) | unsupported | 9.429 (0.739) | unsupported |
| einsum_plan_read_into_accum_f64_s256 | 8.397 (0.155) | unsupported | 13.750 (0.115) | unsupported |
| einsum_plan_strided_read_into_f64_s256 † | 25.595 (0.890) | unsupported | 9.620 (0.242) | unsupported |
| einsum_plan_strided_read_into_accum_f64_s256 † | 28.267 (0.485) | unsupported | 11.439 (0.183) | unsupported |
| sub_strided_read_f64_n1048576 | 34.519 (0.043) | 34.079 (0.052) | 12.113 (0.160) | 60.078 (1.162) |
| div_strided_read_f64_n1048576 | 34.562 (0.036) | 34.129 (0.060) | 12.995 (0.092) | 49.876 (2.523) |
| rem_strided_read_f64_n1048576 | 34.521 (0.070) | 34.094 (0.081) | 12.180 (0.124) | 53.295 (2.057) |
| pow_strided_read_f64_n1048576 | 36.141 (0.064) | 35.876 (0.079) | 29.200 (0.063) | 63.975 (4.272) |
| maximum_strided_read_f64_n1048576 | 34.500 (0.087) | 34.548 (0.085) | 12.073 (0.139) | 66.527 (1.059) |
| minimum_strided_read_f64_n1048576 | 34.562 (0.060) | 34.101 (0.035) | 11.751 (0.267) | 61.625 (1.050) |
| abs_strided_read_f64_n1048576 | 18.926 (0.020) | 18.986 (0.023) | 7.674 (0.059) | 52.620 (1.996) |
| sign_strided_read_f64_n1048576 | 18.992 (0.051) | 19.017 (0.039) | 7.526 (0.025) | 51.867 (2.822) |
| expm1_strided_read_f64_n1048576 | 19.175 (0.024) | 19.213 (0.028) | 9.687 (0.069) | 52.592 (1.645) |
| log_strided_read_f64_n1048576 | 19.449 (0.033) | 19.428 (0.060) | 12.508 (0.057) | 49.012 (2.683) |
| log1p_strided_read_f64_n1048576 | 19.403 (0.023) | 19.454 (0.028) | 12.993 (0.083) | 50.443 (3.481) |
| erf_strided_read_f64_n1048576 | 19.666 (0.025) | 19.729 (0.035) | 14.264 (0.039) | 69.052 (0.997) |
| sin_strided_read_f64_n1048576 | 21.062 (0.050) | 21.289 (0.068) | 10.865 (0.080) | 51.963 (2.189) |
| cos_strided_read_f64_n1048576 | 21.136 (0.049) | 21.247 (0.032) | 10.902 (0.060) | 53.547 (0.810) |
| sqrt_strided_read_f64_n1048576 | 18.971 (0.018) | 19.010 (0.049) | 7.661 (0.053) | 53.193 (1.596) |
| rsqrt_strided_read_f64_n1048576 | 18.986 (0.031) | 18.987 (0.036) | 7.558 (0.058) | 71.698 (1.572) |
| reduce_prod_strided_read_f64_n1048576 † | 35.590 (0.071) | 35.264 (0.058) | 13.616 (0.015) | 52.190 (2.413) |
| reduce_max_strided_read_f64_n1048576 † | 35.901 (0.125) | 36.056 (0.091) | 13.600 (0.006) | 80.160 (3.743) |
| reduce_min_strided_read_f64_n1048576 † | 36.011 (0.101) | 35.401 (0.038) | 13.699 (0.053) | 81.760 (38.853) |
| reduce_sum_squares_strided_read_f64_n1048576 † | 35.951 (0.070) | 35.619 (0.110) | 21.328 (0.035) | 783.164 (8.145) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.684 | 16777216 | 0.221 | 0 |
| abs_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.246 | 16777216 | 1.513 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.606 | 16777216 | 0.613 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.608 | 16777216 | 0.612 | 0 |
| add_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.374 | 25165824 | 1.490 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.415 | 25165824 | 1.342 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.411 | 25165824 | 1.356 | 0 |
| add_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.373 | 25165824 | 1.493 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.412 | 25165824 | 1.353 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.409 | 25165824 | 1.362 | 0 |
| add_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.031 | 25165824 | 0.274 | 0 |
| add_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.389 | 25165824 | 1.431 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.104 | 25165824 | 0.505 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.098 | 25165824 | 0.508 | 0 |
| axpby_read_into_accum_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.491 | 25165824 | 1.134 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.245 | 25165824 | 0.124 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No matching public caller-owned accumulating output API; donation is not output reuse. |
| conj_into_c64_n1048576 | jax-cuda | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.382 | 33554432 | 0.973 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.381 | 33554432 | 0.974 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.381 | 33554432 | 0.976 | 0 |
| conj_read_into_c64_n1048576 | jax-cuda | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.383 | 33554432 | 0.971 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.380 | 33554432 | 0.978 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.380 | 33554432 | 0.977 | 0 |
| conj_strided_read_c64_n1048576 | jax-cuda | 16 | 15 | 0.793 | 33554432 | 0.469 | 0 |
| conj_strided_read_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.383 | 33554432 | 0.970 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.474 | 33554432 | 0.784 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.470 | 33554432 | 0.790 | 0 |
| cos_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.714 | 16777216 | 0.217 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.349 | 16777216 | 1.065 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.676 | 16777216 | 0.549 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.680 | 16777216 | 0.546 | 5.905e-08 |
| div_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.418 | 25165824 | 1.333 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.413 | 25165824 | 1.349 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.407 | 25165824 | 1.370 | 0 |
| div_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.425 | 25165824 | 1.312 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.426 | 25165824 | 1.307 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.405 | 25165824 | 1.376 | 0 |
| div_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.596 | 25165824 | 0.349 | 0 |
| div_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.416 | 25165824 | 1.340 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.106 | 25165824 | 0.504 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.092 | 25165824 | 0.510 | 0 |
| dot_conj_l0r0_c64_n262144 | jax-cuda | 64 | 15 | 3.664 | 8388624 | 0.101 | 1.221e-05 |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | 256 | 15 | 3.218 | 8388624 | 0.462 | 1.366e-05 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | 128 | 15 | 3.108 | 8388624 | 0.239 | 1.212e-05 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | 128 | 15 | 3.100 | 8388624 | 0.240 | 1.212e-05 |
| dot_general_batch_f64_s256 | jax-cuda | 32 | 15 | 2.822 | 3210240 | 0.025 | 7.463e-07 |
| dot_general_batch_f64_s256 | pytorch-cuda | 128 | 15 | 2.429 | 3210240 | 0.117 | 7.463e-07 |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | 166 | 15 | 2.098 | 3210240 | 0.176 | 7.463e-07 |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | 166 | 15 | 2.454 | 3210240 | 0.150 | 7.463e-07 |
| dot_general_read_into_accum_cached_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | 256 | 15 | 2.673 | 2097152 | 0.139 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | 256 | 15 | 2.251 | 2097152 | 0.165 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.642 | 2097152 | 0.141 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | 256 | 15 | 2.249 | 2097152 | 0.165 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.429 | 2097152 | 0.153 | 0 |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | 512 | 15 | 3.846 | 2097152 | 0.193 | 0 |
| dot_general_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| einsum_plan_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.417 | 1070080 | 0.078 | 0 |
| einsum_plan_into_f64_s256 | tenferro-cuda-direct | 498 | 15 | 3.535 | 1070080 | 0.104 | 0 |
| einsum_plan_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_owned_f64_s256 | jax-cuda | 64 | 15 | 3.600 | 1070080 | 0.013 | 0 |
| einsum_plan_owned_f64_s256 | pytorch-cuda | 256 | 15 | 2.702 | 1070080 | 0.070 | 0 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-direct | 256 | 15 | 3.382 | 1070080 | 0.056 | 0 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_f64_s256 | jax-cuda | 64 | 15 | 3.513 | 1070080 | 0.013 | 0 |
| einsum_plan_read_f64_s256 | pytorch-cuda | 256 | 15 | 2.618 | 1070080 | 0.072 | 0 |
| einsum_plan_read_f64_s256 | tenferro-cuda-direct | 256 | 15 | 3.238 | 1070080 | 0.059 | 0 |
| einsum_plan_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_accum_f64_s256 | jax-cuda | — | — | — | 1608704 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 3.520 | 1608704 | 0.081 | 0 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-direct | 256 | 15 | 2.150 | 1608704 | 0.133 | 0 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.414 | 1070080 | 0.079 | 0 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-direct | 498 | 15 | 3.693 | 1070080 | 0.100 | 0 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_f64_s256 | jax-cuda | 64 | 15 | 3.505 | 1070080 | 0.014 | 0 |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | 256 | 15 | 2.552 | 1070080 | 0.074 | 0 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-direct | 64 | 15 | 2.116 | 1070080 | 0.022 | 0 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_accum_f64_s256 | jax-cuda | — | — | — | 1608704 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.928 | 1608704 | 0.097 | 0 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-direct | 128 | 15 | 3.618 | 1608704 | 0.039 | 0 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.463 | 1070080 | 0.077 | 0 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-direct | 128 | 15 | 3.276 | 1070080 | 0.029 | 0 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| erf_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.210 | 16777216 | 0.168 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.456 | 16777216 | 0.814 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.629 | 16777216 | 0.590 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.631 | 16777216 | 0.588 | 8.536e-08 |
| exp_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.663 | 16777216 | 0.223 | 1.095e-07 |
| exp_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.262 | 16777216 | 1.416 | 1.095e-07 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.608 | 16777216 | 0.611 | 1.095e-07 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.611 | 16777216 | 0.608 | 1.095e-07 |
| expm1_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.683 | 16777216 | 0.221 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.310 | 16777216 | 1.199 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.614 | 16777216 | 0.605 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.615 | 16777216 | 0.604 | 7.39e-08 |
| log1p_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.614 | 16777216 | 0.230 | 1.11e-07 |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.416 | 16777216 | 0.894 | 1.11e-07 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.621 | 16777216 | 0.598 | 1.11e-07 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.623 | 16777216 | 0.597 | 1.11e-07 |
| log_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.568 | 16777216 | 0.237 | 4.308e-08 |
| log_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.400 | 16777216 | 0.928 | 4.308e-08 |
| log_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.622 | 16777216 | 0.597 | 4.308e-08 |
| log_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.622 | 16777216 | 0.598 | 4.308e-08 |
| maximum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.129 | 25165824 | 0.262 | 0 |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.386 | 25165824 | 1.442 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.104 | 25165824 | 0.505 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.106 | 25165824 | 0.504 | 0 |
| minimum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.972 | 25165824 | 0.283 | 0 |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.376 | 25165824 | 1.482 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.106 | 25165824 | 0.504 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.091 | 25165824 | 0.511 | 0 |
| mul_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.366 | 25165824 | 1.522 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.411 | 25165824 | 1.356 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.411 | 25165824 | 1.356 | 0 |
| mul_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.383 | 25165824 | 1.454 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.417 | 25165824 | 1.335 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.411 | 25165824 | 1.355 | 0 |
| mul_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.029 | 25165824 | 0.275 | 0 |
| mul_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.379 | 25165824 | 1.472 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.106 | 25165824 | 0.504 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.089 | 25165824 | 0.512 | 0 |
| neg_into_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.241 | 16777216 | 1.541 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.277 | 16777216 | 1.342 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.277 | 16777216 | 1.344 | 0 |
| neg_read_into_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.243 | 16777216 | 1.530 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.277 | 16777216 | 1.342 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.278 | 16777216 | 1.335 | 0 |
| neg_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.934 | 16777216 | 0.192 | 0 |
| neg_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.242 | 16777216 | 1.536 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.607 | 16777216 | 0.612 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.609 | 16777216 | 0.610 | 0 |
| norm_squared_read_f64_n262144 | jax-cuda | 64 | 15 | 3.870 | 2097160 | 0.024 | 1.664e-07 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.376 | 2097160 | 0.156 | 1.664e-07 |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | 256 | 15 | 2.799 | 2097160 | 0.133 | 1.664e-07 |
| norm_squared_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 2097160 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| pow_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.047 | 25165824 | 0.272 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.934 | 25165824 | 0.596 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.157 | 25165824 | 0.482 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.148 | 25165824 | 0.485 | 1.48e-07 |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.565 | 8392704 | 0.072 | 0 |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.482 | 8392704 | 0.427 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.298 | 8392704 | 0.162 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.308 | 8392704 | 0.161 | 0 |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.616 | 8392704 | 0.071 | 0 |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.507 | 8392704 | 0.424 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.305 | 8392704 | 0.161 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.266 | 8392704 | 0.164 | 0 |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 3.340 | 8392704 | 0.111 | 4.995e-06 |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.486 | 8392704 | 0.427 | 4.106e-06 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.278 | 8392704 | 0.163 | 4.661e-06 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.257 | 8392704 | 0.165 | 4.661e-06 |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | 4 | 15 | 3.133 | 8392704 | 0.007 | 3.311e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.730 | 8392704 | 0.272 | 3.309e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.301 | 8392704 | 0.162 | 3.312e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.280 | 8392704 | 0.163 | 3.312e-07 |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.664 | 8392704 | 0.070 | 7.159e-06 |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.492 | 8392704 | 0.426 | 5.935e-06 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.299 | 8392704 | 0.162 | 4.724e-06 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.281 | 8392704 | 0.163 | 4.724e-06 |
| rem_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.705 | 25165824 | 0.327 | 0 |
| rem_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.390 | 25165824 | 1.430 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.105 | 25165824 | 0.504 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.091 | 25165824 | 0.511 | 0 |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.294 | 16777216 | 0.162 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.242 | 16777216 | 1.536 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.608 | 16777216 | 0.612 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.608 | 16777216 | 0.611 | 6.504e-08 |
| sign_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.660 | 16777216 | 0.224 | 0 |
| sign_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.241 | 16777216 | 1.543 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.608 | 16777216 | 0.611 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.609 | 16777216 | 0.611 | 0 |
| sin_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.663 | 16777216 | 0.223 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.348 | 16777216 | 1.069 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.674 | 16777216 | 0.551 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.681 | 16777216 | 0.545 | 4.153e-08 |
| sqrt_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.702 | 16777216 | 0.218 | 0 |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.245 | 16777216 | 1.516 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.607 | 16777216 | 0.612 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.608 | 16777216 | 0.611 | 0 |
| sub_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.369 | 25165824 | 1.512 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.414 | 25165824 | 1.346 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.410 | 25165824 | 1.359 | 0 |
| sub_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.374 | 25165824 | 1.489 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.410 | 25165824 | 1.358 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.412 | 25165824 | 1.354 | 0 |
| sub_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.922 | 25165824 | 0.290 | 0 |
| sub_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.388 | 25165824 | 1.438 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.105 | 25165824 | 0.504 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.091 | 25165824 | 0.511 | 0 |
| tanh_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.599 | 16777216 | 0.232 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.259 | 16777216 | 1.433 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.609 | 16777216 | 0.610 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.611 | 16777216 | 0.608 | 4.437e-08 |
| tensordot_f64_n262144 | jax-cuda | 16 | 15 | 2.057 | 6291456 | 0.034 | 6.714e-06 |
| tensordot_f64_n262144 | pytorch-cuda | 128 | 15 | 3.331 | 6291456 | 0.167 | 6.714e-06 |
| tensordot_f64_n262144 | tenferro-cuda-direct | 128 | 15 | 3.648 | 6291456 | 0.153 | 6.714e-06 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 128 | 15 | 3.692 | 6291456 | 0.151 | 6.714e-06 |
| vdot_read_f64_n262144 | jax-cuda | 64 | 15 | 3.861 | 4194312 | 0.048 | 2.995e-06 |
| vdot_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.490 | 4194312 | 0.298 | 2.995e-06 |
| vdot_read_f64_n262144 | tenferro-cuda-direct | 256 | 15 | 2.885 | 4194312 | 0.258 | 2.995e-06 |
| vdot_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 4194312 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261126_062950/gpu_reuse_contraction.yaml
timestamp: '2026-11-26T06:29:50+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: cc9867403ef65b20357ed79bc78ec5fc2864c4a9
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
  hostname: d73a3a75b493
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
    GPU_RUNTIME_SHA256: a261fcdef53ecfd3885b6d565dc834faef59287ce2da88c3d6754be96741f84d
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
  device_uuid: GPU-e0c82f3d-3a11-a28c-b42e-1c92d26527fd
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
  - 0.093
  - 0.092971
  - 0.09296
  - 0.09182
  - 0.09265
  - 0.09212
  - 0.092231
  - 0.09237
  - 0.09315
  - 0.09669
  - 0.09556
  - 0.092951
  - 0.09288
  - 0.09235
  - 0.09241
  median_ms: 0.09288
  bytes_per_second: 1445065977605.5125
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-26T06:44:52.088020+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-e0c82f3d-3a11-a28c-b42e-1c92d26527fd
  cuda_visible_devices: GPU-e0c82f3d-3a11-a28c-b42e-1c92d26527fd
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
    src/bin/benchmark_gpu_public_api.rs: f807adb35a2c03383fde11227a47dffbf0ec5cc08c45ce298f7d9aa6341bbb67
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: e8121a793ab45e565d150633b7e9a473b4d4f2c457ae36baed344f2c809fcad5
    src/bin/benchmark_gpu_public_api/multi.rs: 950cb1b3c1a7cca37665b58e3de074bd906dafb8c84b17a51c27ce797a73afa8
    src/bin/benchmark_gpu_public_api/plan.rs: 108edbad472e42b1a734552826d6d233085b581a3484cc4444471435160a6fd8
    src/bin/benchmark_gpu_public_api/read_layout.rs: f51757a34c90ce68e6fae0994df74169ed0438022031465349c7379c7b2c85e0
    src/bin/benchmark_gpu_public_api/reuse.rs: 53bee3baaeba93f4e1ee57d0d1b9553c189edd51d1940ea630ad0dbaa05b11eb
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: 88038508963fbc13303fb54b3898808e4303131321cabedb9f38255941ead438
    src/bin/benchmark_gpu_public_api/state.rs: e1cc2781ce7392154f32cc19372c9fc41ae29765aedc544c7ae7bf02f873c12b
    src/bin/benchmark_gpu_public_api/structural.rs: 09b3cbf0dc0579978344c1aa738f6367112a0c25236924441f690a59de3fb80e
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: d1ff68f86496d991bcbc730138fe6e3076001b3f23cc1a94c44b96f6857cb8ed
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 7995bcf409258185e07210969a738896f64860b67df251fcd1100d11128a5431
    data/results/nvidia-gpu/gpu/reuse_contraction/20261126_062950/fixtures/manifest.json: 95c52cff0ebd6b2eb1a4bb3baf78a0a8b1d43d81d2696a49fb694663b5c5586e
    target/release/benchmark_gpu_public_api: 3ff693b2ae7411a94a3e3337550156543b79bbe33a2720b00f7fb8181bdf1fd4
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:45:45Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:45:34Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:45:11Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:45:20Z'
  compile:
    case_wall_seconds: 43.655
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
| add_into_f64_n1048576 | tenferro-cuda-direct | add_into_f64_n1048576 untimed 387.9 ms is 6237% of the 6.2 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | add_read_into_f64_n1048576 untimed 319.4 ms is 5156% of the 6.2 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | tenferro-cuda-direct | sub_into_f64_n1048576 untimed 328.5 ms is 5290% of the 6.2 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | sub_read_into_f64_n1048576 untimed 321.9 ms is 5201% of the 6.2 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | tenferro-cuda-direct | mul_into_f64_n1048576 untimed 323.7 ms is 5075% of the 6.4 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | mul_read_into_f64_n1048576 untimed 303.9 ms is 4874% of the 6.2 ms timed region (budget 20%) |
| div_into_f64_n1048576 | tenferro-cuda-direct | div_into_f64_n1048576 untimed 301.4 ms is 4859% of the 6.2 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | div_read_into_f64_n1048576 untimed 309.6 ms is 4842% of the 6.4 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | tenferro-cuda-direct | neg_into_f64_n1048576 untimed 312.1 ms is 7508% of the 4.2 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | neg_read_into_f64_n1048576 untimed 307.4 ms is 7401% of the 4.2 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | tenferro-cuda-direct | conj_into_c64_n1048576 untimed 314.2 ms is 5489% of the 5.7 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | conj_read_into_c64_n1048576 untimed 307.5 ms is 5393% of the 5.7 ms timed region (budget 20%) |
| tensordot_f64_n262144 | tenferro-cuda-direct | tensordot_f64_n262144 untimed 2522.7 ms is 4544% of the 55.5 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | tenferro-cuda-direct | vdot_read_f64_n262144 untimed 16.8 ms is 39% of the 43.4 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | norm_squared_read_f64_n262144 untimed 11.2 ms is 27% of the 42.3 ms timed region (budget 20%) |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | axpby_read_into_accum_f64_n1048576 untimed 335.5 ms is 990% of the 33.9 ms timed region (budget 20%) |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | dot_general_read_into_f64_s256 untimed 253.5 ms is 438% of the 57.9 ms timed region (budget 20%) |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | dot_general_read_into_accum_f64_s256 untimed 438.9 ms is 1287% of the 34.1 ms timed region (budget 20%) |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | dot_general_read_into_accum_cached_f64_s256 untimed 429.5 ms is 1270% of the 33.8 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | add_strided_read_f64_n1048576 untimed 116.0 ms is 700% of the 16.6 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | mul_strided_read_f64_n1048576 untimed 108.7 ms is 641% of the 16.9 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | neg_strided_read_f64_n1048576 untimed 87.6 ms is 962% of the 9.1 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | exp_strided_read_f64_n1048576 untimed 87.8 ms is 962% of the 9.1 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | tanh_strided_read_f64_n1048576 untimed 102.3 ms is 1119% of the 9.1 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_sum_strided_read_f64_n1048576 untimed 24.4 ms is 71% of the 34.6 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | conj_strided_read_c64_n1048576 untimed 145.3 ms is 2043% of the 7.1 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | dot_general_batch_f64_s256 untimed 45.1 ms is 143% of the 31.6 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | dot_conj_l0r0_c64_n262144 untimed 23.7 ms is 51% of the 46.7 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | tenferro-cuda-direct | einsum_plan_owned_f64_s256 untimed 18.7 ms is 37% of the 50.8 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | tenferro-cuda-direct | einsum_plan_read_f64_s256 untimed 33.2 ms is 68% of the 48.6 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_f64_s256 untimed 26.3 ms is 83% of the 31.7 ms timed region (budget 20%) |
| einsum_plan_into_f64_s256 | tenferro-cuda-direct | einsum_plan_into_f64_s256 untimed 1113.0 ms is 2093% of the 53.2 ms timed region (budget 20%) |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-direct | einsum_plan_read_into_f64_s256 untimed 505.3 ms is 910% of the 55.6 ms timed region (budget 20%) |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-direct | einsum_plan_read_into_accum_f64_s256 untimed 476.3 ms is 1453% of the 32.8 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_into_f64_s256 untimed 434.2 ms is 873% of the 49.7 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_into_accum_f64_s256 untimed 481.7 ms is 888% of the 54.3 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | tenferro-cuda-direct | sub_strided_read_f64_n1048576 untimed 106.7 ms is 644% of the 16.6 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | tenferro-cuda-direct | div_strided_read_f64_n1048576 untimed 96.3 ms is 580% of the 16.6 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | tenferro-cuda-direct | rem_strided_read_f64_n1048576 untimed 96.5 ms is 582% of the 16.6 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | tenferro-cuda-direct | pow_strided_read_f64_n1048576 untimed 103.7 ms is 598% of the 17.4 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-direct | maximum_strided_read_f64_n1048576 untimed 96.2 ms is 580% of the 16.6 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-direct | minimum_strided_read_f64_n1048576 untimed 98.2 ms is 592% of the 16.6 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | tenferro-cuda-direct | abs_strided_read_f64_n1048576 untimed 78.6 ms is 866% of the 9.1 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | tenferro-cuda-direct | sign_strided_read_f64_n1048576 untimed 97.6 ms is 1071% of the 9.1 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-direct | expm1_strided_read_f64_n1048576 untimed 86.7 ms is 942% of the 9.2 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | tenferro-cuda-direct | log_strided_read_f64_n1048576 untimed 85.6 ms is 917% of the 9.3 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-direct | log1p_strided_read_f64_n1048576 untimed 95.6 ms is 1026% of the 9.3 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | tenferro-cuda-direct | erf_strided_read_f64_n1048576 untimed 108.2 ms is 1146% of the 9.4 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | tenferro-cuda-direct | sin_strided_read_f64_n1048576 untimed 91.1 ms is 901% of the 10.1 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | tenferro-cuda-direct | cos_strided_read_f64_n1048576 untimed 83.9 ms is 827% of the 10.1 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | sqrt_strided_read_f64_n1048576 untimed 82.0 ms is 901% of the 9.1 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | rsqrt_strided_read_f64_n1048576 untimed 104.9 ms is 1152% of the 9.1 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_prod_strided_read_f64_n1048576 untimed 20.0 ms is 59% of the 34.2 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_max_strided_read_f64_n1048576 untimed 20.1 ms is 58% of the 34.6 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_min_strided_read_f64_n1048576 untimed 20.6 ms is 59% of the 34.6 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_sum_squares_strided_read_f64_n1048576 untimed 20.1 ms is 58% of the 34.9 ms timed region (budget 20%) |
| add_into_f64_n1048576 | tenferro-cuda-eager | add_into_f64_n1048576 untimed 424.9 ms is 6885% of the 6.2 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | tenferro-cuda-eager | add_read_into_f64_n1048576 untimed 332.7 ms is 5137% of the 6.5 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | tenferro-cuda-eager | sub_into_f64_n1048576 untimed 331.3 ms is 5381% of the 6.2 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | tenferro-cuda-eager | sub_read_into_f64_n1048576 untimed 328.5 ms is 5318% of the 6.2 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | tenferro-cuda-eager | mul_into_f64_n1048576 untimed 347.2 ms is 5649% of the 6.1 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | tenferro-cuda-eager | mul_read_into_f64_n1048576 untimed 356.3 ms is 5755% of the 6.2 ms timed region (budget 20%) |
| div_into_f64_n1048576 | tenferro-cuda-eager | div_into_f64_n1048576 untimed 365.7 ms is 5988% of the 6.1 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | tenferro-cuda-eager | div_read_into_f64_n1048576 untimed 348.7 ms is 5520% of the 6.3 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | tenferro-cuda-eager | neg_into_f64_n1048576 untimed 358.8 ms is 8649% of the 4.1 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | tenferro-cuda-eager | neg_read_into_f64_n1048576 untimed 393.7 ms is 9011% of the 4.4 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | tenferro-cuda-eager | conj_into_c64_n1048576 untimed 373.6 ms is 6399% of the 5.8 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | tenferro-cuda-eager | conj_read_into_c64_n1048576 untimed 381.0 ms is 6680% of the 5.7 ms timed region (budget 20%) |
| tensordot_f64_n262144 | tenferro-cuda-eager | tensordot_f64_n262144 untimed 1829.4 ms is 3300% of the 55.4 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | tenferro-cuda-eager | add_strided_read_f64_n1048576 untimed 80.5 ms is 488% of the 16.5 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | tenferro-cuda-eager | mul_strided_read_f64_n1048576 untimed 86.6 ms is 530% of the 16.3 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | tenferro-cuda-eager | neg_strided_read_f64_n1048576 untimed 72.3 ms is 792% of the 9.1 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | tenferro-cuda-eager | exp_strided_read_f64_n1048576 untimed 64.3 ms is 701% of the 9.2 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-eager | tanh_strided_read_f64_n1048576 untimed 71.8 ms is 782% of the 9.2 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-eager | reduce_sum_strided_read_f64_n1048576 untimed 11.9 ms is 35% of the 34.2 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | tenferro-cuda-eager | conj_strided_read_c64_n1048576 untimed 81.0 ms is 1151% of the 7.0 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | dot_general_batch_f64_s256 untimed 22.2 ms is 59% of the 37.3 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | dot_conj_l0r0_c64_n262144 untimed 13.6 ms is 29% of the 46.5 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | tenferro-cuda-eager | sub_strided_read_f64_n1048576 untimed 81.8 ms is 500% of the 16.4 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | tenferro-cuda-eager | div_strided_read_f64_n1048576 untimed 80.4 ms is 491% of the 16.4 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | tenferro-cuda-eager | rem_strided_read_f64_n1048576 untimed 85.2 ms is 520% of the 16.4 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | tenferro-cuda-eager | pow_strided_read_f64_n1048576 untimed 84.8 ms is 492% of the 17.2 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-eager | maximum_strided_read_f64_n1048576 untimed 84.2 ms is 508% of the 16.6 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-eager | minimum_strided_read_f64_n1048576 untimed 81.0 ms is 495% of the 16.4 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | tenferro-cuda-eager | abs_strided_read_f64_n1048576 untimed 78.0 ms is 856% of the 9.1 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | tenferro-cuda-eager | sign_strided_read_f64_n1048576 untimed 58.5 ms is 624% of the 9.4 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-eager | expm1_strided_read_f64_n1048576 untimed 65.8 ms is 714% of the 9.2 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | tenferro-cuda-eager | log_strided_read_f64_n1048576 untimed 60.1 ms is 645% of the 9.3 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-eager | log1p_strided_read_f64_n1048576 untimed 70.7 ms is 757% of the 9.3 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | tenferro-cuda-eager | erf_strided_read_f64_n1048576 untimed 84.1 ms is 888% of the 9.5 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | tenferro-cuda-eager | sin_strided_read_f64_n1048576 untimed 60.6 ms is 590% of the 10.3 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | tenferro-cuda-eager | cos_strided_read_f64_n1048576 untimed 58.6 ms is 574% of the 10.2 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | sqrt_strided_read_f64_n1048576 untimed 63.5 ms is 696% of the 9.1 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | rsqrt_strided_read_f64_n1048576 untimed 72.7 ms is 796% of the 9.1 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-eager | reduce_prod_strided_read_f64_n1048576 untimed 8.2 ms is 24% of the 34.1 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-eager | reduce_max_strided_read_f64_n1048576 untimed 12.0 ms is 35% of the 34.6 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-eager | reduce_min_strided_read_f64_n1048576 untimed 7.9 ms is 23% of the 34.0 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-eager | reduce_sum_squares_strided_read_f64_n1048576 untimed 8.1 ms is 24% of the 34.2 ms timed region (budget 20%) |
| add_into_f64_n1048576 | pytorch-cuda | add_into_f64_n1048576 untimed 437.2 ms is 7791% of the 5.6 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | pytorch-cuda | add_read_into_f64_n1048576 untimed 282.2 ms is 5049% of the 5.6 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | pytorch-cuda | sub_into_f64_n1048576 untimed 271.4 ms is 4942% of the 5.5 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | pytorch-cuda | sub_read_into_f64_n1048576 untimed 257.9 ms is 4615% of the 5.6 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | pytorch-cuda | mul_into_f64_n1048576 untimed 263.1 ms is 4803% of the 5.5 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | pytorch-cuda | mul_read_into_f64_n1048576 untimed 234.5 ms is 3959% of the 5.9 ms timed region (budget 20%) |
| div_into_f64_n1048576 | pytorch-cuda | div_into_f64_n1048576 untimed 243.8 ms is 3888% of the 6.3 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | pytorch-cuda | div_read_into_f64_n1048576 untimed 173.1 ms is 2722% of the 6.4 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | pytorch-cuda | neg_into_f64_n1048576 untimed 246.9 ms is 6829% of the 3.6 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | pytorch-cuda | neg_read_into_f64_n1048576 untimed 252.0 ms is 6897% of the 3.7 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | pytorch-cuda | conj_into_c64_n1048576 untimed 188.5 ms is 3297% of the 5.7 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | pytorch-cuda | conj_read_into_c64_n1048576 untimed 274.2 ms is 4777% of the 5.7 ms timed region (budget 20%) |
| tensordot_f64_n262144 | pytorch-cuda | tensordot_f64_n262144 untimed 69.3 ms is 140% of the 49.7 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | pytorch-cuda | vdot_read_f64_n262144 untimed 10.6 ms is 28% of the 37.4 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | pytorch-cuda | norm_squared_read_f64_n262144 untimed 7.9 ms is 22% of the 35.6 ms timed region (budget 20%) |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | axpby_read_into_accum_f64_n1048576 untimed 275.5 ms is 3747% of the 7.4 ms timed region (budget 20%) |
| dot_general_read_into_f64_s256 | pytorch-cuda | dot_general_read_into_f64_s256 untimed 35.8 ms is 98% of the 36.6 ms timed region (budget 20%) |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | dot_general_read_into_accum_f64_s256 untimed 60.9 ms is 127% of the 48.0 ms timed region (budget 20%) |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | dot_general_read_into_accum_cached_f64_s256 untimed 35.9 ms is 88% of the 40.7 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | pytorch-cuda | add_strided_read_f64_n1048576 untimed 331.8 ms is 5674% of the 5.8 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | pytorch-cuda | mul_strided_read_f64_n1048576 untimed 288.1 ms is 5115% of the 5.6 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | pytorch-cuda | neg_strided_read_f64_n1048576 untimed 248.2 ms is 6845% of the 3.6 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | pytorch-cuda | exp_strided_read_f64_n1048576 untimed 285.8 ms is 7236% of the 3.9 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | tanh_strided_read_f64_n1048576 untimed 257.3 ms is 6609% of the 3.9 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_strided_read_f64_n1048576 untimed 24.5 ms is 47% of the 52.6 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | pytorch-cuda | conj_strided_read_c64_n1048576 untimed 414.4 ms is 7214% of the 5.7 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | pytorch-cuda | dot_general_batch_f64_s256 untimed 32.3 ms is 89% of the 36.3 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | dot_conj_l0r0_c64_n262144 untimed 11.3 ms is 23% of the 48.4 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | pytorch-cuda | einsum_plan_owned_f64_s256 untimed 12.8 ms is 32% of the 40.5 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | pytorch-cuda | einsum_plan_read_f64_s256 untimed 12.4 ms is 31% of the 39.4 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | einsum_plan_strided_read_f64_s256 untimed 12.0 ms is 31% of the 38.5 ms timed region (budget 20%) |
| einsum_plan_into_f64_s256 | pytorch-cuda | einsum_plan_into_f64_s256 untimed 12.9 ms is 34% of the 38.4 ms timed region (budget 20%) |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | einsum_plan_read_into_f64_s256 untimed 25.8 ms is 69% of the 37.2 ms timed region (budget 20%) |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_read_into_accum_f64_s256 untimed 39.9 ms is 76% of the 52.6 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_f64_s256 untimed 18.8 ms is 51% of the 37.0 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_accum_f64_s256 untimed 40.9 ms is 93% of the 44.0 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | pytorch-cuda | sub_strided_read_f64_n1048576 untimed 289.5 ms is 4969% of the 5.8 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | pytorch-cuda | div_strided_read_f64_n1048576 untimed 300.0 ms is 4803% of the 6.2 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | pytorch-cuda | rem_strided_read_f64_n1048576 untimed 298.0 ms is 5099% of the 5.8 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | pytorch-cuda | pow_strided_read_f64_n1048576 untimed 313.8 ms is 2238% of the 14.0 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | maximum_strided_read_f64_n1048576 untimed 273.9 ms is 4717% of the 5.8 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | minimum_strided_read_f64_n1048576 untimed 156.9 ms is 2793% of the 5.6 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | pytorch-cuda | abs_strided_read_f64_n1048576 untimed 184.4 ms is 5013% of the 3.7 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | pytorch-cuda | sign_strided_read_f64_n1048576 untimed 231.8 ms is 6405% of the 3.6 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | expm1_strided_read_f64_n1048576 untimed 249.5 ms is 5333% of the 4.7 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | pytorch-cuda | log_strided_read_f64_n1048576 untimed 241.3 ms is 4016% of the 6.0 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | log1p_strided_read_f64_n1048576 untimed 259.3 ms is 4142% of the 6.3 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | pytorch-cuda | erf_strided_read_f64_n1048576 untimed 259.4 ms is 3784% of the 6.9 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | pytorch-cuda | sin_strided_read_f64_n1048576 untimed 234.4 ms is 4485% of the 5.2 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | pytorch-cuda | cos_strided_read_f64_n1048576 untimed 253.8 ms is 4848% of the 5.2 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | sqrt_strided_read_f64_n1048576 untimed 258.8 ms is 7025% of the 3.7 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | rsqrt_strided_read_f64_n1048576 untimed 257.1 ms is 7036% of the 3.7 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | reduce_prod_strided_read_f64_n1048576 untimed 741.6 ms is 1418% of the 52.3 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | reduce_max_strided_read_f64_n1048576 untimed 24.2 ms is 46% of the 52.2 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | reduce_min_strided_read_f64_n1048576 untimed 26.9 ms is 51% of the 53.1 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_squares_strided_read_f64_n1048576 untimed 35.2 ms is 86% of the 41.0 ms timed region (budget 20%) |
| tensordot_f64_n262144 | jax-cuda | tensordot_f64_n262144 untimed 728.4 ms is 1909% of the 38.2 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | jax-cuda | vdot_read_f64_n262144 untimed 152.6 ms is 262% of the 58.1 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | jax-cuda | norm_squared_read_f64_n262144 untimed 13.1 ms is 23% of the 57.9 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | jax-cuda | add_strided_read_f64_n1048576 untimed 345.1 ms is 1135% of the 30.4 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | jax-cuda | mul_strided_read_f64_n1048576 untimed 197.7 ms is 654% of the 30.2 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | jax-cuda | neg_strided_read_f64_n1048576 untimed 162.4 ms is 559% of the 29.1 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | jax-cuda | exp_strided_read_f64_n1048576 untimed 195.7 ms is 781% of the 25.1 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | jax-cuda | tanh_strided_read_f64_n1048576 untimed 195.8 ms is 813% of the 24.1 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | reduce_sum_strided_read_f64_n1048576 untimed 179.6 ms is 466% of the 38.6 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | jax-cuda | conj_strided_read_c64_n1048576 untimed 350.4 ms is 2932% of the 12.0 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | jax-cuda | dot_general_batch_f64_s256 untimed 576.1 ms is 1344% of the 42.9 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | jax-cuda | dot_conj_l0r0_c64_n262144 untimed 83.7 ms is 150% of the 55.8 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | jax-cuda | einsum_plan_owned_f64_s256 untimed 460.7 ms is 853% of the 54.0 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | jax-cuda | einsum_plan_read_f64_s256 untimed 14.8 ms is 28% of the 53.6 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | jax-cuda | einsum_plan_strided_read_f64_s256 untimed 211.1 ms is 398% of the 53.0 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | jax-cuda | sub_strided_read_f64_n1048576 untimed 188.8 ms is 656% of the 28.8 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | jax-cuda | div_strided_read_f64_n1048576 untimed 188.9 ms is 788% of the 24.0 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | jax-cuda | rem_strided_read_f64_n1048576 untimed 192.7 ms is 751% of the 25.7 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | jax-cuda | pow_strided_read_f64_n1048576 untimed 236.3 ms is 752% of the 31.4 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | jax-cuda | maximum_strided_read_f64_n1048576 untimed 239.1 ms is 745% of the 32.1 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | jax-cuda | minimum_strided_read_f64_n1048576 untimed 181.7 ms is 610% of the 29.8 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | jax-cuda | abs_strided_read_f64_n1048576 untimed 179.5 ms is 705% of the 25.5 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | jax-cuda | sign_strided_read_f64_n1048576 untimed 176.8 ms is 701% of the 25.2 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | jax-cuda | expm1_strided_read_f64_n1048576 untimed 178.6 ms is 703% of the 25.4 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | jax-cuda | log_strided_read_f64_n1048576 untimed 184.8 ms is 784% of the 23.6 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | jax-cuda | log1p_strided_read_f64_n1048576 untimed 195.4 ms is 810% of the 24.1 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | jax-cuda | erf_strided_read_f64_n1048576 untimed 168.5 ms is 505% of the 33.3 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | jax-cuda | sin_strided_read_f64_n1048576 untimed 243.7 ms is 984% of the 24.8 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | jax-cuda | cos_strided_read_f64_n1048576 untimed 211.9 ms is 823% of the 25.8 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | jax-cuda | sqrt_strided_read_f64_n1048576 untimed 218.1 ms is 786% of the 27.8 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | rsqrt_strided_read_f64_n1048576 untimed 187.8 ms is 547% of the 34.4 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | reduce_prod_strided_read_f64_n1048576 untimed 159.5 ms is 315% of the 50.6 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | reduce_max_strided_read_f64_n1048576 untimed 214.7 ms is 567% of the 37.9 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | reduce_min_strided_read_f64_n1048576 untimed 107.3 ms is 255% of the 42.1 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | reduce_sum_squares_strided_read_f64_n1048576 untimed 175.4 ms is 373% of the 47.0 ms timed region (budget 20%) |