# gpu/reuse_contraction: CUDA public API benchmark

Measured cells: **189**. Unsupported cells: **35**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1417.4 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261128_082346/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**17** of 56) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

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
| add_into_f64_n1048576 | 13.799 (0.194) | 13.654 (0.280) | 12.172 (0.194) | unsupported |
| add_read_into_f64_n1048576 | 13.683 (0.310) | 13.413 (0.164) | 11.929 (0.156) | unsupported |
| sub_into_f64_n1048576 | 13.524 (0.208) | 13.547 (0.310) | 12.048 (0.121) | unsupported |
| sub_read_into_f64_n1048576 | 13.302 (0.351) | 13.435 (0.403) | 12.248 (0.165) | unsupported |
| mul_into_f64_n1048576 | 13.656 (0.324) | 13.363 (0.220) | 12.097 (0.184) | unsupported |
| mul_read_into_f64_n1048576 | 13.634 (0.371) | 13.676 (0.349) | 12.368 (0.124) | unsupported |
| div_into_f64_n1048576 | 13.429 (0.343) | 13.361 (0.472) | 13.451 (0.151) | unsupported |
| div_read_into_f64_n1048576 | 13.621 (0.286) | 13.506 (0.264) | 13.272 (0.105) | unsupported |
| neg_into_f64_n1048576 | 8.716 (0.057) | 8.754 (0.025) | 7.696 (0.058) | unsupported |
| neg_read_into_f64_n1048576 | 8.725 (0.040) | 8.782 (0.040) | 7.835 (0.146) | unsupported |
| conj_into_c64_n1048576 | 25.139 (0.218) | 25.117 (0.125) | 25.229 (0.334) | unsupported |
| conj_read_into_c64_n1048576 | 24.949 (0.222) | 25.211 (0.143) | 25.306 (0.164) | unsupported |
| tensordot_f64_n262144 † | 28.538 (0.297) | 28.411 (0.508) | 25.050 (0.213) | 98.276 (7.724) |
| vdot_read_f64_n262144 † | 11.280 (0.007) | unsupported | 9.768 (0.156) | 37.676 (2.100) |
| norm_squared_read_f64_n262144 † | 10.825 (0.019) | unsupported | 9.266 (0.015) | 38.164 (1.212) |
| axpby_read_into_accum_f64_n1048576 † | 124.923 (1.132) | unsupported | 16.527 (0.115) | unsupported |
| dot_general_read_into_f64_s256 | 7.520 (0.005) | unsupported | 7.573 (0.012) | unsupported |
| dot_general_read_into_accum_f64_s256 | 8.835 (0.011) | unsupported | 8.912 (0.011) | unsupported |
| dot_general_read_into_accum_cached_f64_s256 | 8.828 (0.015) | unsupported | 8.909 (0.021) | unsupported |
| add_strided_read_f64_n1048576 | 34.669 (0.056) | 34.629 (0.032) | 12.467 (0.172) | 35.141 (1.753) |
| mul_strided_read_f64_n1048576 | 34.669 (0.084) | 34.449 (0.061) | 12.078 (0.250) | 40.964 (2.126) |
| neg_strided_read_f64_n1048576 | 18.946 (0.041) | 19.101 (0.052) | 7.786 (0.159) | 44.438 (1.583) |
| exp_strided_read_f64_n1048576 | 19.043 (0.026) | 19.172 (0.023) | 8.201 (0.068) | 42.317 (2.536) |
| tanh_strided_read_f64_n1048576 | 19.038 (0.023) | 19.194 (0.060) | 8.154 (0.027) | 41.691 (2.637) |
| reduce_sum_strided_read_f64_n1048576 † | 36.759 (0.100) | 36.227 (0.064) | 13.574 (0.018) | 43.110 (1.592) |
| conj_strided_read_c64_n1048576 | 30.516 (0.134) | 30.082 (0.491) | 25.374 (0.353) | 39.227 (2.044) |
| dot_general_batch_f64_s256 † | 11.887 (0.019) | 11.998 (0.030) | 16.740 (0.085) | 64.068 (2.404) |
| dot_conj_l0r0_c64_n262144 † | 24.395 (0.034) | 24.414 (0.047) | 12.572 (0.012) | 38.816 (1.739) |
| einsum_plan_owned_f64_s256 † | 9.480 (0.245) | unsupported | 7.841 (0.103) | 38.339 (1.530) |
| einsum_plan_read_f64_s256 † | 9.702 (0.239) | unsupported | 7.796 (0.184) | 38.282 (2.147) |
| einsum_plan_strided_read_f64_s256 † | 22.454 (0.918) | unsupported | 7.800 (0.212) | 35.597 (1.839) |
| einsum_plan_into_f64_s256 | 6.042 (0.006) | unsupported | 7.692 (0.014) | unsupported |
| einsum_plan_read_into_f64_s256 | 6.038 (0.007) | unsupported | 7.677 (0.018) | unsupported |
| einsum_plan_read_into_accum_f64_s256 † | 7.332 (0.002) | unsupported | 8.882 (0.039) | unsupported |
| einsum_plan_strided_read_into_f64_s256 † | 18.181 (0.113) | unsupported | 7.592 (0.017) | unsupported |
| einsum_plan_strided_read_into_accum_f64_s256 † | 19.508 (0.036) | unsupported | 8.774 (0.062) | unsupported |
| sub_strided_read_f64_n1048576 | 34.673 (0.080) | 34.450 (0.104) | 12.439 (0.123) | 37.898 (3.554) |
| div_strided_read_f64_n1048576 | 34.770 (0.039) | 34.472 (0.094) | 13.310 (0.152) | 35.179 (1.966) |
| rem_strided_read_f64_n1048576 | 34.721 (0.088) | 34.434 (0.130) | 12.632 (0.203) | 36.987 (1.250) |
| pow_strided_read_f64_n1048576 | 36.227 (0.068) | 36.158 (0.177) | 29.286 (0.075) | 39.859 (2.076) |
| maximum_strided_read_f64_n1048576 | 34.718 (0.067) | 34.368 (0.115) | 12.688 (0.135) | 43.880 (2.790) |
| minimum_strided_read_f64_n1048576 | 34.730 (0.063) | 34.342 (0.047) | 12.322 (0.086) | 39.878 (1.744) |
| abs_strided_read_f64_n1048576 | 18.939 (0.083) | 19.070 (0.027) | 7.759 (0.045) | 32.299 (1.925) |
| sign_strided_read_f64_n1048576 | 18.963 (0.038) | 19.095 (0.058) | 7.616 (0.024) | 31.430 (1.259) |
| expm1_strided_read_f64_n1048576 | 19.251 (0.035) | 19.351 (0.038) | 9.741 (0.067) | 32.101 (1.318) |
| log_strided_read_f64_n1048576 | 19.487 (0.031) | 19.560 (0.035) | 12.497 (0.020) | 33.478 (1.390) |
| log1p_strided_read_f64_n1048576 | 19.457 (0.023) | 19.589 (0.015) | 13.043 (0.057) | 47.422 (2.589) |
| erf_strided_read_f64_n1048576 | 19.810 (0.057) | 19.845 (0.059) | 14.342 (0.040) | 60.292 (1.747) |
| sin_strided_read_f64_n1048576 | 21.165 (0.056) | 21.431 (0.039) | 10.937 (0.046) | 44.858 (2.241) |
| cos_strided_read_f64_n1048576 | 21.112 (0.032) | 21.382 (0.047) | 10.987 (0.037) | 44.976 (1.953) |
| sqrt_strided_read_f64_n1048576 | 18.995 (0.036) | 19.092 (0.022) | 7.768 (0.088) | 44.785 (3.156) |
| rsqrt_strided_read_f64_n1048576 | 18.961 (0.020) | 19.080 (0.056) | 7.674 (0.050) | 60.309 (1.733) |
| reduce_prod_strided_read_f64_n1048576 † | 36.492 (0.041) | 36.011 (0.080) | 13.525 (0.008) | 46.403 (1.496) |
| reduce_max_strided_read_f64_n1048576 † | 36.936 (0.060) | 36.919 (0.073) | 13.627 (0.021) | 46.637 (1.637) |
| reduce_min_strided_read_f64_n1048576 † | 36.840 (0.046) | 36.391 (0.043) | 13.685 (0.014) | 47.297 (2.264) |
| reduce_sum_squares_strided_read_f64_n1048576 † | 36.841 (0.098) | 36.380 (0.048) | 21.255 (0.026) | 672.855 (12.806) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.034 | 16777216 | 0.366 | 0 |
| abs_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.248 | 16777216 | 1.526 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.606 | 16777216 | 0.625 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.610 | 16777216 | 0.621 | 0 |
| add_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.389 | 25165824 | 1.459 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.442 | 25165824 | 1.287 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.437 | 25165824 | 1.300 | 0 |
| add_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.382 | 25165824 | 1.488 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.438 | 25165824 | 1.298 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.429 | 25165824 | 1.324 | 0 |
| add_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.125 | 25165824 | 0.505 | 0 |
| add_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.399 | 25165824 | 1.424 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.109 | 25165824 | 0.512 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.108 | 25165824 | 0.513 | 0 |
| axpby_read_into_accum_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.529 | 25165824 | 1.074 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.999 | 25165824 | 0.142 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No matching public caller-owned accumulating output API; donation is not output reuse. |
| conj_into_c64_n1048576 | jax-cuda | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.404 | 33554432 | 0.938 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.402 | 33554432 | 0.942 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.402 | 33554432 | 0.943 | 0 |
| conj_read_into_c64_n1048576 | jax-cuda | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.405 | 33554432 | 0.935 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.399 | 33554432 | 0.949 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.403 | 33554432 | 0.939 | 0 |
| conj_strided_read_c64_n1048576 | jax-cuda | 16 | 15 | 0.628 | 33554432 | 0.603 | 0 |
| conj_strided_read_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.406 | 33554432 | 0.933 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.488 | 33554432 | 0.776 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.481 | 33554432 | 0.787 | 0 |
| cos_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.439 | 16777216 | 0.263 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.352 | 16777216 | 1.077 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.676 | 16777216 | 0.561 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.684 | 16777216 | 0.554 | 5.905e-08 |
| div_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.430 | 25165824 | 1.320 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.430 | 25165824 | 1.322 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.428 | 25165824 | 1.329 | 0 |
| div_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.425 | 25165824 | 1.338 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.436 | 25165824 | 1.303 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.432 | 25165824 | 1.315 | 0 |
| div_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.126 | 25165824 | 0.505 | 0 |
| div_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.426 | 25165824 | 1.334 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.113 | 25165824 | 0.511 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.103 | 25165824 | 0.515 | 0 |
| dot_conj_l0r0_c64_n262144 | jax-cuda | 64 | 15 | 2.484 | 8388624 | 0.152 | 1.221e-05 |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | 256 | 15 | 3.218 | 8388624 | 0.471 | 1.366e-05 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | 128 | 15 | 3.123 | 8388624 | 0.243 | 1.212e-05 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | 128 | 15 | 3.125 | 8388624 | 0.242 | 1.212e-05 |
| dot_general_batch_f64_s256 | jax-cuda | 32 | 15 | 2.050 | 3210240 | 0.035 | 0 |
| dot_general_batch_f64_s256 | pytorch-cuda | 128 | 15 | 2.143 | 3210240 | 0.135 | 0 |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | 166 | 15 | 1.973 | 3210240 | 0.191 | 0 |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | 166 | 15 | 1.992 | 3210240 | 0.189 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | 256 | 15 | 2.281 | 2097152 | 0.166 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | 256 | 15 | 2.260 | 2097152 | 0.168 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.281 | 2097152 | 0.166 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | 256 | 15 | 2.262 | 2097152 | 0.167 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | pytorch-cuda | 512 | 15 | 3.878 | 2097152 | 0.195 | 0 |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | 512 | 15 | 3.850 | 2097152 | 0.197 | 0 |
| dot_general_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| einsum_plan_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_into_f64_s256 | pytorch-cuda | 498 | 15 | 3.830 | 1070080 | 0.098 | 1.093e-06 |
| einsum_plan_into_f64_s256 | tenferro-cuda-direct | 498 | 15 | 3.009 | 1070080 | 0.125 | 1.093e-06 |
| einsum_plan_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_owned_f64_s256 | jax-cuda | 64 | 15 | 2.454 | 1070080 | 0.020 | 1.093e-06 |
| einsum_plan_owned_f64_s256 | pytorch-cuda | 256 | 15 | 2.007 | 1070080 | 0.096 | 1.093e-06 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-direct | 256 | 15 | 2.427 | 1070080 | 0.080 | 1.093e-06 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_f64_s256 | jax-cuda | 64 | 15 | 2.450 | 1070080 | 0.020 | 1.093e-06 |
| einsum_plan_read_f64_s256 | pytorch-cuda | 256 | 15 | 1.996 | 1070080 | 0.097 | 1.093e-06 |
| einsum_plan_read_f64_s256 | tenferro-cuda-direct | 256 | 15 | 2.484 | 1070080 | 0.078 | 1.093e-06 |
| einsum_plan_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_accum_f64_s256 | jax-cuda | — | — | — | 1608704 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.274 | 1608704 | 0.128 | 3.261e-07 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-direct | 498 | 15 | 3.651 | 1608704 | 0.155 | 3.261e-07 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | 498 | 15 | 3.823 | 1070080 | 0.098 | 1.093e-06 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-direct | 498 | 15 | 3.007 | 1070080 | 0.125 | 1.093e-06 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_f64_s256 | jax-cuda | 64 | 15 | 2.278 | 1070080 | 0.021 | 1.093e-06 |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | 256 | 15 | 1.997 | 1070080 | 0.097 | 1.093e-06 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.874 | 1070080 | 0.034 | 1.093e-06 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_accum_f64_s256 | jax-cuda | — | — | — | 1608704 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.246 | 1608704 | 0.129 | 3.261e-07 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.497 | 1608704 | 0.058 | 3.261e-07 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | 498 | 15 | 3.781 | 1070080 | 0.099 | 1.093e-06 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.327 | 1070080 | 0.042 | 1.093e-06 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| erf_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.929 | 16777216 | 0.196 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.459 | 16777216 | 0.825 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.634 | 16777216 | 0.597 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.635 | 16777216 | 0.596 | 8.536e-08 |
| exp_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.354 | 16777216 | 0.280 | 1.093e-07 |
| exp_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.262 | 16777216 | 1.443 | 1.093e-07 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.609 | 16777216 | 0.622 | 1.093e-07 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.614 | 16777216 | 0.617 | 1.093e-07 |
| expm1_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.027 | 16777216 | 0.369 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.312 | 16777216 | 1.215 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.616 | 16777216 | 0.615 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.619 | 16777216 | 0.612 | 7.39e-08 |
| log1p_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.517 | 16777216 | 0.250 | 1.106e-07 |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.417 | 16777216 | 0.908 | 1.106e-07 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.623 | 16777216 | 0.608 | 1.106e-07 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.627 | 16777216 | 0.604 | 1.106e-07 |
| log_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.071 | 16777216 | 0.354 | 4.308e-08 |
| log_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.400 | 16777216 | 0.947 | 4.308e-08 |
| log_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.624 | 16777216 | 0.607 | 4.308e-08 |
| log_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.626 | 16777216 | 0.605 | 4.308e-08 |
| maximum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.404 | 25165824 | 0.405 | 0 |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.406 | 25165824 | 1.399 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.111 | 25165824 | 0.511 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.100 | 25165824 | 0.517 | 0 |
| minimum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.276 | 25165824 | 0.445 | 0 |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.394 | 25165824 | 1.441 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.111 | 25165824 | 0.511 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.099 | 25165824 | 0.517 | 0 |
| mul_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.387 | 25165824 | 1.468 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.437 | 25165824 | 1.300 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.428 | 25165824 | 1.329 | 0 |
| mul_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.396 | 25165824 | 1.436 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.436 | 25165824 | 1.302 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.438 | 25165824 | 1.298 | 0 |
| mul_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.311 | 25165824 | 0.433 | 0 |
| mul_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.386 | 25165824 | 1.470 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.109 | 25165824 | 0.512 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.102 | 25165824 | 0.515 | 0 |
| neg_into_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.246 | 16777216 | 1.538 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.279 | 16777216 | 1.358 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.280 | 16777216 | 1.352 | 0 |
| neg_read_into_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.251 | 16777216 | 1.511 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.279 | 16777216 | 1.357 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.281 | 16777216 | 1.348 | 0 |
| neg_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.422 | 16777216 | 0.266 | 0 |
| neg_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.249 | 16777216 | 1.520 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.606 | 16777216 | 0.625 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.611 | 16777216 | 0.620 | 0 |
| norm_squared_read_f64_n262144 | jax-cuda | 64 | 15 | 2.442 | 2097160 | 0.039 | 4.991e-07 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.372 | 2097160 | 0.160 | 4.991e-07 |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | 256 | 15 | 2.771 | 2097160 | 0.137 | 4.991e-07 |
| norm_squared_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 2097160 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| pow_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.276 | 25165824 | 0.445 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.937 | 25165824 | 0.606 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.159 | 25165824 | 0.490 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.157 | 25165824 | 0.491 | 1.48e-07 |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 2.985 | 8392704 | 0.127 | 0 |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.489 | 8392704 | 0.435 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.364 | 8392704 | 0.160 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.363 | 8392704 | 0.160 | 0 |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 3.027 | 8392704 | 0.125 | 0 |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.503 | 8392704 | 0.433 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.358 | 8392704 | 0.161 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.329 | 8392704 | 0.163 | 0 |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 2.970 | 8392704 | 0.128 | 4.995e-06 |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.462 | 8392704 | 0.438 | 4.106e-06 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.335 | 8392704 | 0.162 | 4.661e-06 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.305 | 8392704 | 0.164 | 4.661e-06 |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | 4 | 15 | 2.691 | 8392704 | 0.009 | 3.311e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.721 | 8392704 | 0.279 | 3.309e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.358 | 8392704 | 0.161 | 3.312e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.328 | 8392704 | 0.163 | 3.312e-07 |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 2.759 | 8392704 | 0.137 | 7.159e-06 |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.475 | 8392704 | 0.436 | 5.935e-06 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.353 | 8392704 | 0.161 | 4.724e-06 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.319 | 8392704 | 0.163 | 4.724e-06 |
| rem_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.184 | 25165824 | 0.480 | 0 |
| rem_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.404 | 25165824 | 1.406 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.111 | 25165824 | 0.511 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.102 | 25165824 | 0.516 | 0 |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.930 | 16777216 | 0.196 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.246 | 16777216 | 1.542 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.607 | 16777216 | 0.624 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.611 | 16777216 | 0.620 | 6.504e-08 |
| sign_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.006 | 16777216 | 0.377 | 0 |
| sign_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.244 | 16777216 | 1.554 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.607 | 16777216 | 0.624 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.611 | 16777216 | 0.620 | 0 |
| sin_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.435 | 16777216 | 0.264 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.350 | 16777216 | 1.082 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.677 | 16777216 | 0.559 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.686 | 16777216 | 0.552 | 4.153e-08 |
| sqrt_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.433 | 16777216 | 0.264 | 0 |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.249 | 16777216 | 1.524 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.608 | 16777216 | 0.623 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.611 | 16777216 | 0.620 | 0 |
| sub_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.386 | 25165824 | 1.474 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.433 | 25165824 | 1.313 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.433 | 25165824 | 1.311 | 0 |
| sub_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.392 | 25165824 | 1.450 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.426 | 25165824 | 1.335 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.430 | 25165824 | 1.322 | 0 |
| sub_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.213 | 25165824 | 0.468 | 0 |
| sub_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.398 | 25165824 | 1.427 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.110 | 25165824 | 0.512 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.102 | 25165824 | 0.515 | 0 |
| tanh_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.334 | 16777216 | 0.284 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.261 | 16777216 | 1.452 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.609 | 16777216 | 0.622 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.614 | 16777216 | 0.617 | 4.437e-08 |
| tensordot_f64_n262144 | jax-cuda | 32 | 15 | 3.145 | 6291456 | 0.045 | 6.714e-06 |
| tensordot_f64_n262144 | pytorch-cuda | 128 | 15 | 3.206 | 6291456 | 0.177 | 6.714e-06 |
| tensordot_f64_n262144 | tenferro-cuda-direct | 128 | 15 | 3.653 | 6291456 | 0.156 | 6.714e-06 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 128 | 15 | 3.637 | 6291456 | 0.156 | 6.714e-06 |
| vdot_read_f64_n262144 | jax-cuda | 64 | 15 | 2.411 | 4194312 | 0.079 | 2.995e-06 |
| vdot_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.500 | 4194312 | 0.303 | 2.995e-06 |
| vdot_read_f64_n262144 | tenferro-cuda-direct | 256 | 15 | 2.888 | 4194312 | 0.262 | 2.995e-06 |
| vdot_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 4194312 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261128_082346/gpu_reuse_contraction.yaml
timestamp: '2026-11-28T08:23:46+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: 28b35e0408e272b66d7460e3fa12213799c543f9
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
  hostname: 32b786430f5d
  os: Linux-6.8.0-149-generic-x86_64-with-glibc2.39
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
  device_uuid: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
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
  - 0.092901
  - 0.093931
  - 0.096375
  - 0.094431
  - 0.094271
  - 0.095394
  - 0.095072
  - 0.09288
  - 0.095744
  - 0.094693
  - 0.09337
  - 0.096606
  - 0.095193
  - 0.09309
  - 0.094954
  median_ms: 0.094693
  bytes_per_second: 1417398625030.3613
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-28T08:36:50.478359+00:00'
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
    data/results/nvidia-gpu/gpu/reuse_contraction/20261128_082346/fixtures/manifest.json: 95c52cff0ebd6b2eb1a4bb3baf78a0a8b1d43d81d2696a49fb694663b5c5586e
    target/release/benchmark_gpu_public_api: 54dda9d9f7540806d7ff5a5d68e1312bc33b0d58871d8e3e9467a819e7c59318
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-28T08:37:34Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-28T08:37:22Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-28T08:37:06Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-28T08:37:14Z'
  compile:
    case_wall_seconds: 35.121
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
| add_into_f64_n1048576 | tenferro-cuda-direct | add_into_f64_n1048576 untimed 345.5 ms is 5224% of the 6.6 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | add_read_into_f64_n1048576 untimed 285.8 ms is 4363% of the 6.6 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | tenferro-cuda-direct | sub_into_f64_n1048576 untimed 353.7 ms is 5464% of the 6.5 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | sub_read_into_f64_n1048576 untimed 262.7 ms is 4104% of the 6.4 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | tenferro-cuda-direct | mul_into_f64_n1048576 untimed 273.0 ms is 4184% of the 6.5 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | mul_read_into_f64_n1048576 untimed 281.4 ms is 4314% of the 6.5 ms timed region (budget 20%) |
| div_into_f64_n1048576 | tenferro-cuda-direct | div_into_f64_n1048576 untimed 274.5 ms is 4268% of the 6.4 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | div_read_into_f64_n1048576 untimed 261.9 ms is 4003% of the 6.5 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | tenferro-cuda-direct | neg_into_f64_n1048576 untimed 290.3 ms is 6923% of the 4.2 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | neg_read_into_f64_n1048576 untimed 262.1 ms is 6232% of the 4.2 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | tenferro-cuda-direct | conj_into_c64_n1048576 untimed 272.9 ms is 4526% of the 6.0 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | conj_read_into_c64_n1048576 untimed 299.5 ms is 5000% of the 6.0 ms timed region (budget 20%) |
| tensordot_f64_n262144 | tenferro-cuda-direct | tensordot_f64_n262144 untimed 2010.3 ms is 3666% of the 54.8 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | tenferro-cuda-direct | vdot_read_f64_n262144 untimed 42.1 ms is 97% of the 43.3 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | norm_squared_read_f64_n262144 untimed 10.4 ms is 25% of the 41.6 ms timed region (budget 20%) |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | axpby_read_into_accum_f64_n1048576 untimed 288.9 ms is 964% of the 30.0 ms timed region (budget 20%) |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | dot_general_read_into_f64_s256 untimed 245.7 ms is 425% of the 57.8 ms timed region (budget 20%) |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | dot_general_read_into_accum_f64_s256 untimed 259.5 ms is 765% of the 33.9 ms timed region (budget 20%) |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | dot_general_read_into_accum_cached_f64_s256 untimed 198.8 ms is 585% of the 34.0 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | add_strided_read_f64_n1048576 untimed 88.8 ms is 534% of the 16.6 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | mul_strided_read_f64_n1048576 untimed 90.8 ms is 546% of the 16.6 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | neg_strided_read_f64_n1048576 untimed 74.2 ms is 814% of the 9.1 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | exp_strided_read_f64_n1048576 untimed 79.5 ms is 870% of the 9.1 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | tanh_strided_read_f64_n1048576 untimed 78.3 ms is 857% of the 9.1 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_sum_strided_read_f64_n1048576 untimed 15.6 ms is 44% of the 35.3 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | conj_strided_read_c64_n1048576 untimed 125.6 ms is 1719% of the 7.3 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | dot_general_batch_f64_s256 untimed 25.3 ms is 85% of the 29.6 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | dot_conj_l0r0_c64_n262144 untimed 16.9 ms is 36% of the 46.9 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | tenferro-cuda-direct | einsum_plan_owned_f64_s256 untimed 12.7 ms is 34% of the 37.0 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | tenferro-cuda-direct | einsum_plan_read_f64_s256 untimed 12.4 ms is 33% of the 37.7 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_f64_s256 untimed 12.5 ms is 29% of the 43.1 ms timed region (budget 20%) |
| einsum_plan_into_f64_s256 | tenferro-cuda-direct | einsum_plan_into_f64_s256 untimed 383.2 ms is 847% of the 45.3 ms timed region (budget 20%) |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-direct | einsum_plan_read_into_f64_s256 untimed 386.0 ms is 855% of the 45.1 ms timed region (budget 20%) |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-direct | einsum_plan_read_into_accum_f64_s256 untimed 446.4 ms is 815% of the 54.8 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_into_f64_s256 untimed 389.8 ms is 1114% of the 35.0 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-direct | einsum_plan_strided_read_into_accum_f64_s256 untimed 386.5 ms is 1019% of the 37.9 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | tenferro-cuda-direct | sub_strided_read_f64_n1048576 untimed 90.3 ms is 543% of the 16.6 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | tenferro-cuda-direct | div_strided_read_f64_n1048576 untimed 85.8 ms is 514% of the 16.7 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | tenferro-cuda-direct | rem_strided_read_f64_n1048576 untimed 93.9 ms is 563% of the 16.7 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | tenferro-cuda-direct | pow_strided_read_f64_n1048576 untimed 88.5 ms is 509% of the 17.4 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-direct | maximum_strided_read_f64_n1048576 untimed 85.4 ms is 513% of the 16.7 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-direct | minimum_strided_read_f64_n1048576 untimed 85.4 ms is 512% of the 16.7 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | tenferro-cuda-direct | abs_strided_read_f64_n1048576 untimed 84.8 ms is 933% of the 9.1 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | tenferro-cuda-direct | sign_strided_read_f64_n1048576 untimed 88.3 ms is 967% of the 9.1 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-direct | expm1_strided_read_f64_n1048576 untimed 78.1 ms is 845% of the 9.2 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | tenferro-cuda-direct | log_strided_read_f64_n1048576 untimed 77.6 ms is 830% of the 9.4 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-direct | log1p_strided_read_f64_n1048576 untimed 87.3 ms is 935% of the 9.3 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | tenferro-cuda-direct | erf_strided_read_f64_n1048576 untimed 123.6 ms is 1298% of the 9.5 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | tenferro-cuda-direct | sin_strided_read_f64_n1048576 untimed 92.4 ms is 909% of the 10.2 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | tenferro-cuda-direct | cos_strided_read_f64_n1048576 untimed 77.5 ms is 764% of the 10.1 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | sqrt_strided_read_f64_n1048576 untimed 76.4 ms is 838% of the 9.1 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | rsqrt_strided_read_f64_n1048576 untimed 85.2 ms is 936% of the 9.1 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_prod_strided_read_f64_n1048576 untimed 19.9 ms is 57% of the 35.0 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_max_strided_read_f64_n1048576 untimed 19.7 ms is 56% of the 35.5 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_min_strided_read_f64_n1048576 untimed 17.4 ms is 49% of the 35.4 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-direct | reduce_sum_squares_strided_read_f64_n1048576 untimed 17.5 ms is 50% of the 35.4 ms timed region (budget 20%) |
| add_into_f64_n1048576 | tenferro-cuda-eager | add_into_f64_n1048576 untimed 378.0 ms is 5784% of the 6.5 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | tenferro-cuda-eager | add_read_into_f64_n1048576 untimed 304.6 ms is 4708% of the 6.5 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | tenferro-cuda-eager | sub_into_f64_n1048576 untimed 292.5 ms is 4515% of the 6.5 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | tenferro-cuda-eager | sub_read_into_f64_n1048576 untimed 302.6 ms is 4683% of the 6.5 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | tenferro-cuda-eager | mul_into_f64_n1048576 untimed 297.4 ms is 4613% of the 6.4 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | tenferro-cuda-eager | mul_read_into_f64_n1048576 untimed 270.9 ms is 4159% of the 6.5 ms timed region (budget 20%) |
| div_into_f64_n1048576 | tenferro-cuda-eager | div_into_f64_n1048576 untimed 290.2 ms is 4500% of the 6.5 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | tenferro-cuda-eager | div_read_into_f64_n1048576 untimed 278.0 ms is 4307% of the 6.5 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | tenferro-cuda-eager | neg_into_f64_n1048576 untimed 268.2 ms is 6380% of the 4.2 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | tenferro-cuda-eager | neg_read_into_f64_n1048576 untimed 302.3 ms is 7159% of the 4.2 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | tenferro-cuda-eager | conj_into_c64_n1048576 untimed 278.7 ms is 4620% of the 6.0 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | tenferro-cuda-eager | conj_read_into_c64_n1048576 untimed 275.8 ms is 4557% of the 6.1 ms timed region (budget 20%) |
| tensordot_f64_n262144 | tenferro-cuda-eager | tensordot_f64_n262144 untimed 2036.2 ms is 3479% of the 58.5 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | tenferro-cuda-eager | add_strided_read_f64_n1048576 untimed 73.4 ms is 441% of the 16.6 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | tenferro-cuda-eager | mul_strided_read_f64_n1048576 untimed 74.4 ms is 450% of the 16.6 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | tenferro-cuda-eager | neg_strided_read_f64_n1048576 untimed 64.2 ms is 700% of the 9.2 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | tenferro-cuda-eager | exp_strided_read_f64_n1048576 untimed 55.2 ms is 599% of the 9.2 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-eager | tanh_strided_read_f64_n1048576 untimed 58.6 ms is 634% of the 9.2 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-eager | reduce_sum_strided_read_f64_n1048576 untimed 11.9 ms is 34% of the 34.8 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | tenferro-cuda-eager | conj_strided_read_c64_n1048576 untimed 66.1 ms is 915% of the 7.2 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | dot_general_batch_f64_s256 untimed 17.8 ms is 59% of the 29.9 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | dot_conj_l0r0_c64_n262144 untimed 10.1 ms is 22% of the 46.9 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | tenferro-cuda-eager | sub_strided_read_f64_n1048576 untimed 68.6 ms is 414% of the 16.5 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | tenferro-cuda-eager | div_strided_read_f64_n1048576 untimed 73.1 ms is 442% of the 16.5 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | tenferro-cuda-eager | rem_strided_read_f64_n1048576 untimed 74.5 ms is 451% of the 16.5 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | tenferro-cuda-eager | pow_strided_read_f64_n1048576 untimed 80.6 ms is 464% of the 17.4 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-eager | maximum_strided_read_f64_n1048576 untimed 74.2 ms is 449% of the 16.5 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-eager | minimum_strided_read_f64_n1048576 untimed 76.3 ms is 463% of the 16.5 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | tenferro-cuda-eager | abs_strided_read_f64_n1048576 untimed 67.2 ms is 734% of the 9.2 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | tenferro-cuda-eager | sign_strided_read_f64_n1048576 untimed 53.6 ms is 585% of the 9.2 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-eager | expm1_strided_read_f64_n1048576 untimed 55.2 ms is 594% of the 9.3 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | tenferro-cuda-eager | log_strided_read_f64_n1048576 untimed 54.2 ms is 578% of the 9.4 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-eager | log1p_strided_read_f64_n1048576 untimed 58.7 ms is 624% of the 9.4 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | tenferro-cuda-eager | erf_strided_read_f64_n1048576 untimed 64.6 ms is 678% of the 9.5 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | tenferro-cuda-eager | sin_strided_read_f64_n1048576 untimed 53.8 ms is 523% of the 10.3 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | tenferro-cuda-eager | cos_strided_read_f64_n1048576 untimed 54.7 ms is 533% of the 10.3 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | sqrt_strided_read_f64_n1048576 untimed 54.0 ms is 589% of the 9.2 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | rsqrt_strided_read_f64_n1048576 untimed 60.8 ms is 663% of the 9.2 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-eager | reduce_prod_strided_read_f64_n1048576 untimed 7.8 ms is 22% of the 34.6 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-eager | reduce_max_strided_read_f64_n1048576 untimed 7.7 ms is 22% of the 35.4 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-eager | reduce_min_strided_read_f64_n1048576 untimed 7.7 ms is 22% of the 34.9 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-eager | reduce_sum_squares_strided_read_f64_n1048576 untimed 7.7 ms is 22% of the 34.9 ms timed region (budget 20%) |
| add_into_f64_n1048576 | pytorch-cuda | add_into_f64_n1048576 untimed 168.8 ms is 2895% of the 5.8 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | pytorch-cuda | add_read_into_f64_n1048576 untimed 85.0 ms is 1488% of the 5.7 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | pytorch-cuda | sub_into_f64_n1048576 untimed 55.1 ms is 954% of the 5.8 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | pytorch-cuda | sub_read_into_f64_n1048576 untimed 53.4 ms is 907% of the 5.9 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | pytorch-cuda | mul_into_f64_n1048576 untimed 58.7 ms is 1010% of the 5.8 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | pytorch-cuda | mul_read_into_f64_n1048576 untimed 50.7 ms is 855% of the 5.9 ms timed region (budget 20%) |
| div_into_f64_n1048576 | pytorch-cuda | div_into_f64_n1048576 untimed 61.6 ms is 953% of the 6.5 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | pytorch-cuda | div_read_into_f64_n1048576 untimed 52.2 ms is 816% of the 6.4 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | pytorch-cuda | neg_into_f64_n1048576 untimed 65.7 ms is 1783% of the 3.7 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | pytorch-cuda | neg_read_into_f64_n1048576 untimed 56.6 ms is 1504% of the 3.8 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | pytorch-cuda | conj_into_c64_n1048576 untimed 76.1 ms is 1256% of the 6.1 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | pytorch-cuda | conj_read_into_c64_n1048576 untimed 76.5 ms is 1262% of the 6.1 ms timed region (budget 20%) |
| tensordot_f64_n262144 | pytorch-cuda | tensordot_f64_n262144 untimed 54.2 ms is 112% of the 48.5 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | pytorch-cuda | vdot_read_f64_n262144 untimed 9.3 ms is 25% of the 37.7 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | pytorch-cuda | norm_squared_read_f64_n262144 untimed 7.6 ms is 21% of the 35.6 ms timed region (budget 20%) |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | axpby_read_into_accum_f64_n1048576 untimed 65.5 ms is 827% of the 7.9 ms timed region (budget 20%) |
| dot_general_read_into_f64_s256 | pytorch-cuda | dot_general_read_into_f64_s256 untimed 105.7 ms is 182% of the 58.2 ms timed region (budget 20%) |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | dot_general_read_into_accum_f64_s256 untimed 46.1 ms is 135% of the 34.2 ms timed region (budget 20%) |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | dot_general_read_into_accum_cached_f64_s256 untimed 29.4 ms is 86% of the 34.3 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | pytorch-cuda | add_strided_read_f64_n1048576 untimed 72.9 ms is 1215% of the 6.0 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | pytorch-cuda | mul_strided_read_f64_n1048576 untimed 79.1 ms is 1368% of the 5.8 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | pytorch-cuda | neg_strided_read_f64_n1048576 untimed 188.8 ms is 4939% of the 3.8 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | pytorch-cuda | exp_strided_read_f64_n1048576 untimed 245.4 ms is 6222% of the 3.9 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | tanh_strided_read_f64_n1048576 untimed 220.7 ms is 5639% of the 3.9 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_strided_read_f64_n1048576 untimed 23.5 ms is 45% of the 52.1 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | pytorch-cuda | conj_strided_read_c64_n1048576 untimed 310.0 ms is 5083% of the 6.1 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | pytorch-cuda | dot_general_batch_f64_s256 untimed 31.3 ms is 97% of the 32.2 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | dot_conj_l0r0_c64_n262144 untimed 11.4 ms is 22% of the 51.8 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | pytorch-cuda | einsum_plan_owned_f64_s256 untimed 13.3 ms is 44% of the 30.2 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | pytorch-cuda | einsum_plan_read_f64_s256 untimed 10.4 ms is 35% of the 30.2 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | einsum_plan_strided_read_f64_s256 untimed 10.5 ms is 35% of the 30.0 ms timed region (budget 20%) |
| einsum_plan_into_f64_s256 | pytorch-cuda | einsum_plan_into_f64_s256 untimed 13.7 ms is 24% of the 57.5 ms timed region (budget 20%) |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | einsum_plan_read_into_f64_s256 untimed 105.6 ms is 182% of the 58.0 ms timed region (budget 20%) |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_read_into_accum_f64_s256 untimed 35.0 ms is 102% of the 34.2 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_f64_s256 untimed 13.9 ms is 25% of the 56.7 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_accum_f64_s256 untimed 33.0 ms is 98% of the 33.8 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | pytorch-cuda | sub_strided_read_f64_n1048576 untimed 228.3 ms is 3815% of the 6.0 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | pytorch-cuda | div_strided_read_f64_n1048576 untimed 78.9 ms is 1233% of the 6.4 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | pytorch-cuda | rem_strided_read_f64_n1048576 untimed 104.1 ms is 1716% of the 6.1 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | pytorch-cuda | pow_strided_read_f64_n1048576 untimed 114.1 ms is 812% of the 14.1 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | maximum_strided_read_f64_n1048576 untimed 96.8 ms is 1586% of the 6.1 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | minimum_strided_read_f64_n1048576 untimed 73.8 ms is 1249% of the 5.9 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | pytorch-cuda | abs_strided_read_f64_n1048576 untimed 60.1 ms is 1615% of the 3.7 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | pytorch-cuda | sign_strided_read_f64_n1048576 untimed 57.3 ms is 1557% of the 3.7 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | expm1_strided_read_f64_n1048576 untimed 61.1 ms is 1305% of the 4.7 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | pytorch-cuda | log_strided_read_f64_n1048576 untimed 66.8 ms is 1112% of the 6.0 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | log1p_strided_read_f64_n1048576 untimed 61.4 ms is 980% of the 6.3 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | pytorch-cuda | erf_strided_read_f64_n1048576 untimed 65.9 ms is 956% of the 6.9 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | pytorch-cuda | sin_strided_read_f64_n1048576 untimed 59.9 ms is 1140% of the 5.3 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | pytorch-cuda | cos_strided_read_f64_n1048576 untimed 64.0 ms is 1209% of the 5.3 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | sqrt_strided_read_f64_n1048576 untimed 56.0 ms is 1498% of the 3.7 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | rsqrt_strided_read_f64_n1048576 untimed 60.5 ms is 1641% of the 3.7 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | reduce_prod_strided_read_f64_n1048576 untimed 663.0 ms is 1276% of the 51.9 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | reduce_max_strided_read_f64_n1048576 untimed 17.1 ms is 33% of the 52.3 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | reduce_min_strided_read_f64_n1048576 untimed 18.4 ms is 35% of the 52.5 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_squares_strided_read_f64_n1048576 untimed 17.2 ms is 42% of the 40.8 ms timed region (budget 20%) |
| tensordot_f64_n262144 | jax-cuda | tensordot_f64_n262144 untimed 615.6 ms is 1263% of the 48.7 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | jax-cuda | vdot_read_f64_n262144 untimed 162.7 ms is 443% of the 36.7 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | jax-cuda | norm_squared_read_f64_n262144 untimed 9.7 ms is 26% of the 36.6 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | jax-cuda | add_strided_read_f64_n1048576 untimed 474.7 ms is 2811% of the 16.9 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | jax-cuda | mul_strided_read_f64_n1048576 untimed 348.6 ms is 1774% of the 19.6 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | jax-cuda | neg_strided_read_f64_n1048576 untimed 226.6 ms is 1066% of the 21.3 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | jax-cuda | exp_strided_read_f64_n1048576 untimed 331.9 ms is 1655% of the 20.1 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | jax-cuda | tanh_strided_read_f64_n1048576 untimed 329.0 ms is 1674% of the 19.6 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | reduce_sum_strided_read_f64_n1048576 untimed 137.6 ms is 329% of the 41.8 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | jax-cuda | conj_strided_read_c64_n1048576 untimed 456.7 ms is 4752% of the 9.6 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | jax-cuda | dot_general_batch_f64_s256 untimed 549.3 ms is 1777% of the 30.9 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | jax-cuda | dot_conj_l0r0_c64_n262144 untimed 75.4 ms is 202% of the 37.3 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | jax-cuda | einsum_plan_owned_f64_s256 untimed 387.6 ms is 1061% of the 36.5 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | jax-cuda | einsum_plan_read_f64_s256 untimed 12.0 ms is 33% of the 36.3 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | jax-cuda | einsum_plan_strided_read_f64_s256 untimed 175.7 ms is 509% of the 34.5 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | jax-cuda | sub_strided_read_f64_n1048576 untimed 335.0 ms is 1741% of the 19.2 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | jax-cuda | div_strided_read_f64_n1048576 untimed 335.8 ms is 1966% of the 17.1 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | jax-cuda | rem_strided_read_f64_n1048576 untimed 324.6 ms is 1824% of the 17.8 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | jax-cuda | pow_strided_read_f64_n1048576 untimed 338.1 ms is 1765% of the 19.2 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | jax-cuda | maximum_strided_read_f64_n1048576 untimed 216.0 ms is 1027% of the 21.0 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | jax-cuda | minimum_strided_read_f64_n1048576 untimed 317.7 ms is 1679% of the 18.9 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | jax-cuda | abs_strided_read_f64_n1048576 untimed 229.2 ms is 1476% of the 15.5 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | jax-cuda | sign_strided_read_f64_n1048576 untimed 318.7 ms is 2092% of the 15.2 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | jax-cuda | expm1_strided_read_f64_n1048576 untimed 334.1 ms is 2163% of the 15.4 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | jax-cuda | log_strided_read_f64_n1048576 untimed 276.2 ms is 1730% of the 16.0 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | jax-cuda | log1p_strided_read_f64_n1048576 untimed 182.9 ms is 805% of the 22.7 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | jax-cuda | erf_strided_read_f64_n1048576 untimed 169.6 ms is 586% of the 28.9 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | jax-cuda | sin_strided_read_f64_n1048576 untimed 177.3 ms is 832% of the 21.3 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | jax-cuda | cos_strided_read_f64_n1048576 untimed 190.3 ms is 870% of the 21.9 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | jax-cuda | sqrt_strided_read_f64_n1048576 untimed 157.2 ms is 718% of the 21.9 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | rsqrt_strided_read_f64_n1048576 untimed 161.2 ms is 553% of the 29.2 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | reduce_prod_strided_read_f64_n1048576 untimed 118.4 ms is 257% of the 46.0 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | reduce_max_strided_read_f64_n1048576 untimed 173.6 ms is 386% of the 45.0 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | reduce_min_strided_read_f64_n1048576 untimed 77.3 ms is 160% of the 48.4 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | reduce_sum_squares_strided_read_f64_n1048576 untimed 169.2 ms is 421% of the 40.2 ms timed region (budget 20%) |