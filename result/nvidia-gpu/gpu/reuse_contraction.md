# gpu/reuse_contraction: CUDA public API benchmark

Measured cells: **94**. Unsupported cells: **74**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1410.5 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261108_090627/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**13** of 56) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Transposed borrowed-read cases use prepared native CUDA views in tenferro and PyTorch. JAX compares the same logical values in its native representation; its columns do not claim an equivalent public strided-view contract. View creation and input conversion are outside timing.

Generalized batch dot preserves native output order: tenferro [m,n,b], PyTorch/JAX [b,m,n]. Input permutation views and dimension-number configurations are prepared before timing. Output axes are normalized only on downloaded CPU arrays for verification; no timed output transpose or copy is added.

Prepared einsum uses the public ConcreteEinsumPlan for ik,kj->ij, built once outside execution and sampling. Native matrix multiplication/reuse/accumulation references match the selected mathematical and output-ownership contracts; they do not assert an equivalent prepared-plan interface. Transposed input views/values, read/write handles and accumulation seed restoration are untimed.

| Case | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: |
| add_into_f64_n1048576 | unsupported | 12.100 (0.163) | unsupported |
| add_read_into_f64_n1048576 | unsupported | 11.770 (0.151) | unsupported |
| sub_into_f64_n1048576 | unsupported | 11.947 (0.276) | unsupported |
| sub_read_into_f64_n1048576 | unsupported | 12.056 (0.134) | unsupported |
| mul_into_f64_n1048576 | unsupported | 12.086 (0.203) | unsupported |
| mul_read_into_f64_n1048576 | unsupported | 12.764 (0.184) | unsupported |
| div_into_f64_n1048576 | unsupported | 13.776 (0.173) | unsupported |
| div_read_into_f64_n1048576 | unsupported | 13.769 (0.171) | unsupported |
| neg_into_f64_n1048576 | unsupported | 7.654 (0.160) | unsupported |
| neg_read_into_f64_n1048576 | unsupported | 7.818 (0.145) | unsupported |
| conj_into_c64_n1048576 | unsupported | 25.315 (0.296) | unsupported |
| conj_read_into_c64_n1048576 | unsupported | 25.230 (0.283) | unsupported |
| tensordot_f64_n262144 † | 46.793 (3.884) | 25.298 (0.698) | 95.156 (2.339) |
| vdot_read_f64_n262144 † | unsupported | 9.297 (0.036) | 39.748 (2.091) |
| norm_squared_read_f64_n262144 † | unsupported | 9.022 (0.017) | 39.828 (1.247) |
| axpby_read_into_accum_f64_n1048576 | unsupported | 16.398 (0.201) | unsupported |
| dot_general_read_into_f64_s256 | unsupported | 7.509 (0.036) | unsupported |
| dot_general_read_into_accum_f64_s256 | unsupported | 8.856 (0.014) | unsupported |
| dot_general_read_into_accum_cached_f64_s256 | unsupported | 8.860 (0.022) | unsupported |
| add_strided_read_f64_n1048576 | unsupported | 12.375 (0.086) | 44.529 (2.209) |
| mul_strided_read_f64_n1048576 | unsupported | 12.065 (0.239) | 42.447 (1.636) |
| neg_strided_read_f64_n1048576 | unsupported | 7.688 (0.085) | 35.699 (2.297) |
| exp_strided_read_f64_n1048576 | unsupported | 8.121 (0.069) | 45.647 (1.679) |
| tanh_strided_read_f64_n1048576 | unsupported | 8.127 (0.036) | 47.668 (1.913) |
| reduce_sum_strided_read_f64_n1048576 † | unsupported | 13.531 (0.015) | 49.141 (1.890) |
| conj_strided_read_c64_n1048576 | unsupported | 25.341 (0.275) | 63.361 (5.944) |
| dot_general_batch_f64_s256 † | 23.831 (0.364) | 16.577 (0.153) | 93.145 (1.815) |
| dot_conj_l0r0_c64_n262144 † | 33.845 (0.203) | 12.509 (0.036) | 72.827 (1.560) |
| einsum_plan_owned_f64_s256 † | unsupported | 11.128 (3.836) | 67.179 (3.065) |
| einsum_plan_read_f64_s256 † | unsupported | 7.878 (0.186) | 68.902 (0.742) |
| einsum_plan_strided_read_f64_s256 † | unsupported | 7.860 (0.320) | 66.773 (1.199) |
| einsum_plan_into_f64_s256 | unsupported | 7.613 (0.045) | unsupported |
| einsum_plan_read_into_f64_s256 | unsupported | 7.584 (0.019) | unsupported |
| einsum_plan_read_into_accum_f64_s256 | unsupported | 8.805 (0.056) | unsupported |
| einsum_plan_strided_read_into_f64_s256 | unsupported | 7.505 (0.021) | unsupported |
| einsum_plan_strided_read_into_accum_f64_s256 | unsupported | 8.667 (0.071) | unsupported |
| sub_strided_read_f64_n1048576 | unsupported | 12.433 (0.180) | 72.725 (2.532) |
| div_strided_read_f64_n1048576 | unsupported | 13.235 (0.144) | 69.986 (0.410) |
| rem_strided_read_f64_n1048576 | unsupported | 12.455 (0.173) | 34.816 (2.019) |
| pow_strided_read_f64_n1048576 | unsupported | 29.255 (0.114) | 36.566 (1.952) |
| maximum_strided_read_f64_n1048576 | unsupported | 12.525 (0.137) | 41.079 (3.151) |
| minimum_strided_read_f64_n1048576 | unsupported | 12.270 (0.136) | 41.714 (2.487) |
| abs_strided_read_f64_n1048576 | unsupported | 7.672 (0.082) | 33.248 (2.014) |
| sign_strided_read_f64_n1048576 | unsupported | 7.564 (0.076) | 33.388 (1.669) |
| expm1_strided_read_f64_n1048576 | unsupported | 9.802 (0.222) | 33.770 (2.046) |
| log_strided_read_f64_n1048576 | unsupported | 12.484 (0.042) | 34.113 (1.545) |
| log1p_strided_read_f64_n1048576 | unsupported | 12.986 (0.062) | 32.936 (1.932) |
| erf_strided_read_f64_n1048576 | unsupported | 14.312 (0.089) | 45.538 (2.874) |
| sin_strided_read_f64_n1048576 | unsupported | 10.929 (0.058) | 32.978 (1.548) |
| cos_strided_read_f64_n1048576 | unsupported | 11.003 (0.102) | 34.854 (2.875) |
| sqrt_strided_read_f64_n1048576 | unsupported | 7.756 (0.124) | 33.648 (2.646) |
| rsqrt_strided_read_f64_n1048576 | unsupported | 7.749 (0.159) | 49.530 (1.243) |
| reduce_prod_strided_read_f64_n1048576 † | unsupported | 13.502 (0.017) | 37.451 (1.454) |
| reduce_max_strided_read_f64_n1048576 † | unsupported | 13.621 (0.019) | 69.518 (1.271) |
| reduce_min_strided_read_f64_n1048576 † | unsupported | 13.642 (0.011) | 67.734 (10.004) |
| reduce_sum_squares_strided_read_f64_n1048576 † | unsupported | 21.164 (0.043) | 828.576 (7.460) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.064 | 16777216 | 0.358 | 0 |
| abs_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.246 | 16777216 | 1.550 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| add_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.387 | 25165824 | 1.474 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.377 | 25165824 | 1.516 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.425 | 25165824 | 0.401 | 0 |
| add_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.396 | 25165824 | 1.442 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| axpby_read_into_accum_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.525 | 25165824 | 1.088 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No matching public caller-owned accumulating output API; donation is not output reuse. |
| conj_into_c64_n1048576 | jax-cuda | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.405 | 33554432 | 0.940 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1048576 | jax-cuda | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.404 | 33554432 | 0.943 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_strided_read_c64_n1048576 | jax-cuda | 16 | 15 | 1.014 | 33554432 | 0.375 | 0 |
| conj_strided_read_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.405 | 33554432 | 0.939 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554432 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| cos_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.115 | 16777216 | 0.341 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.352 | 16777216 | 1.081 | 5.905e-08 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| div_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.441 | 25165824 | 1.295 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.441 | 25165824 | 1.296 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.240 | 25165824 | 0.255 | 0 |
| div_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.424 | 25165824 | 1.348 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| dot_conj_l0r0_c64_n262144 | jax-cuda | 32 | 15 | 2.330 | 8388624 | 0.082 | 1.221e-05 |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | 256 | 15 | 3.202 | 8388624 | 0.475 | 1.366e-05 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | 64 | 15 | 2.166 | 8388624 | 0.176 | 1.212e-05 |
| dot_general_batch_f64_s256 | jax-cuda | 32 | 15 | 2.981 | 3210240 | 0.024 | 0 |
| dot_general_batch_f64_s256 | pytorch-cuda | 128 | 15 | 2.122 | 3210240 | 0.137 | 0 |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | 64 | 15 | 1.525 | 3210240 | 0.095 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | 256 | 15 | 2.268 | 2097152 | 0.168 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.267 | 2097152 | 0.168 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | jax-cuda | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | pytorch-cuda | 512 | 15 | 3.845 | 2097152 | 0.198 | 0 |
| dot_general_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| einsum_plan_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_into_f64_s256 | pytorch-cuda | 498 | 15 | 3.791 | 1070080 | 0.100 | 1.093e-06 |
| einsum_plan_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_owned_f64_s256 | jax-cuda | 32 | 15 | 2.150 | 1070080 | 0.011 | 1.093e-06 |
| einsum_plan_owned_f64_s256 | pytorch-cuda | 256 | 15 | 2.849 | 1070080 | 0.068 | 1.093e-06 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_f64_s256 | jax-cuda | 32 | 15 | 2.205 | 1070080 | 0.011 | 1.093e-06 |
| einsum_plan_read_f64_s256 | pytorch-cuda | 256 | 15 | 2.017 | 1070080 | 0.096 | 1.093e-06 |
| einsum_plan_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_accum_f64_s256 | jax-cuda | — | — | — | 1608704 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.254 | 1608704 | 0.130 | 3.261e-07 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | 498 | 15 | 3.777 | 1070080 | 0.100 | 1.093e-06 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_f64_s256 | jax-cuda | 32 | 15 | 2.137 | 1070080 | 0.011 | 1.093e-06 |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | 256 | 15 | 2.012 | 1070080 | 0.097 | 1.093e-06 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_accum_f64_s256 | jax-cuda | — | — | — | 1608704 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.219 | 1608704 | 0.132 | 3.261e-07 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | 1608704 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_f64_s256 | jax-cuda | — | — | — | 1070080 | unavailable | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | 498 | 15 | 3.738 | 1070080 | 0.101 | 1.093e-06 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | 1070080 | unavailable | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| erf_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.457 | 16777216 | 0.261 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.458 | 16777216 | 0.831 | 8.536e-08 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| exp_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.461 | 16777216 | 0.261 | 1.093e-07 |
| exp_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.260 | 16777216 | 1.465 | 1.093e-07 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| expm1_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.081 | 16777216 | 0.352 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.314 | 16777216 | 1.213 | 7.39e-08 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log1p_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.054 | 16777216 | 0.361 | 1.106e-07 |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.416 | 16777216 | 0.916 | 1.106e-07 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.092 | 16777216 | 0.349 | 4.308e-08 |
| log_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.399 | 16777216 | 0.953 | 4.308e-08 |
| log_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| maximum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.315 | 25165824 | 0.434 | 0 |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.401 | 25165824 | 1.424 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| minimum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.335 | 25165824 | 0.428 | 0 |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.393 | 25165824 | 1.454 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| mul_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.387 | 25165824 | 1.476 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.408 | 25165824 | 1.398 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.358 | 25165824 | 0.420 | 0 |
| mul_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.386 | 25165824 | 1.479 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| neg_into_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.245 | 16777216 | 1.554 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.250 | 16777216 | 1.521 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.142 | 16777216 | 0.333 | 0 |
| neg_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.246 | 16777216 | 1.547 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| norm_squared_read_f64_n262144 | jax-cuda | 64 | 15 | 2.549 | 2097160 | 0.037 | 4.991e-07 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.310 | 2097160 | 0.165 | 4.991e-07 |
| norm_squared_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 2097160 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| pow_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.170 | 25165824 | 0.488 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.936 | 25165824 | 0.610 | 1.48e-07 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 4.449 | 8392704 | 0.086 | 0 |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.487 | 8392704 | 0.437 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.167 | 8392704 | 0.088 | 0 |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.492 | 8392704 | 0.436 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 2.397 | 8392704 | 0.159 | 4.995e-06 |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.457 | 8392704 | 0.441 | 4.106e-06 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | 4 | 15 | 3.314 | 8392704 | 0.007 | 3.311e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.709 | 8392704 | 0.281 | 3.309e-07 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 3.145 | 8392704 | 0.121 | 7.159e-06 |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.464 | 8392704 | 0.440 | 5.935e-06 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8392704 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rem_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.114 | 25165824 | 0.512 | 0 |
| rem_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.399 | 25165824 | 1.432 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.585 | 16777216 | 0.240 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.248 | 16777216 | 1.535 | 6.504e-08 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sign_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.068 | 16777216 | 0.356 | 0 |
| sign_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.242 | 16777216 | 1.572 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sin_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.055 | 16777216 | 0.361 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.350 | 16777216 | 1.088 | 4.153e-08 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sqrt_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.077 | 16777216 | 0.353 | 0 |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.248 | 16777216 | 1.533 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sub_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.382 | 25165824 | 1.493 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1048576 | jax-cuda | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.386 | 25165824 | 1.480 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.327 | 25165824 | 0.245 | 0 |
| sub_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.398 | 25165824 | 1.435 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 25165824 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tanh_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.525 | 16777216 | 0.250 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.260 | 16777216 | 1.464 | 4.437e-08 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tensordot_f64_n262144 | jax-cuda | 32 | 15 | 3.045 | 6291456 | 0.047 | 6.714e-06 |
| tensordot_f64_n262144 | pytorch-cuda | 128 | 15 | 3.238 | 6291456 | 0.176 | 6.714e-06 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 8 | 15 | 0.374 | 6291456 | 0.095 | 6.714e-06 |
| vdot_read_f64_n262144 | jax-cuda | 64 | 15 | 2.544 | 4194312 | 0.075 | 2.995e-06 |
| vdot_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.380 | 4194312 | 0.320 | 2.995e-06 |
| vdot_read_f64_n262144 | tenferro-cuda-eager | — | — | — | 4194312 | unavailable | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261108_090627/gpu_reuse_contraction.yaml
timestamp: '2026-11-08T09:06:27+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: 44b5fe8b1b4b339b68f9e9f7dd56d03f6cb08303
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
  hostname: 053e3b59f698
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
  - 0.094503
  - 0.092109
  - 0.096365
  - 0.094383
  - 0.092979
  - 0.096314
  - 0.095204
  - 0.092058
  - 0.096244
  - 0.096185
  - 0.092869
  - 0.096406
  - 0.095274
  - 0.09341
  - 0.095153
  median_ms: 0.095153
  bytes_per_second: 1410546467268.5044
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-08T09:19:27.016206+00:00'
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
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 4f94c67a046ec57f95823fd01f3f73a710d97745fe6a58a3c6106dfec870d65c
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 788b9fde4f8ca4162921baef1203b2e31bb747ec30327a425de3442cf872df52
    data/results/nvidia-gpu/gpu/reuse_contraction/20261108_090627/fixtures/manifest.json: 706a6a5d725bf440a03748d8ba95eeff194de50a81b8547ec97f56312b6629ef
    target/release/benchmark_gpu_public_api: 62f92e0c72d6fb46531ee8106f112567dfd32ddb482255a7cc41ba6819929786
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T09:20:53Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T09:20:26Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T09:19:46Z'
  compile:
    case_wall_seconds: 65.191
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
| tensordot_f64_n262144 | tenferro-cuda-eager | tensordot_f64_n262144 untimed 2260.6 ms is 40561% of the 5.6 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | dot_general_batch_f64_s256 untimed 12.4 ms is 54% of the 22.9 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | dot_conj_l0r0_c64_n262144 untimed 16.9 ms is 52% of the 32.5 ms timed region (budget 20%) |
| add_into_f64_n1048576 | pytorch-cuda | add_into_f64_n1048576 untimed 1011.0 ms is 17409% of the 5.8 ms timed region (budget 20%) |
| add_read_into_f64_n1048576 | pytorch-cuda | add_read_into_f64_n1048576 untimed 899.2 ms is 15943% of the 5.6 ms timed region (budget 20%) |
| sub_into_f64_n1048576 | pytorch-cuda | sub_into_f64_n1048576 untimed 868.5 ms is 15112% of the 5.7 ms timed region (budget 20%) |
| sub_read_into_f64_n1048576 | pytorch-cuda | sub_read_into_f64_n1048576 untimed 867.0 ms is 14995% of the 5.8 ms timed region (budget 20%) |
| mul_into_f64_n1048576 | pytorch-cuda | mul_into_f64_n1048576 untimed 900.8 ms is 15562% of the 5.8 ms timed region (budget 20%) |
| mul_read_into_f64_n1048576 | pytorch-cuda | mul_read_into_f64_n1048576 untimed 1343.3 ms is 21892% of the 6.1 ms timed region (budget 20%) |
| div_into_f64_n1048576 | pytorch-cuda | div_into_f64_n1048576 untimed 1346.2 ms is 20263% of the 6.6 ms timed region (budget 20%) |
| div_read_into_f64_n1048576 | pytorch-cuda | div_read_into_f64_n1048576 untimed 1326.2 ms is 20050% of the 6.6 ms timed region (budget 20%) |
| neg_into_f64_n1048576 | pytorch-cuda | neg_into_f64_n1048576 untimed 885.4 ms is 23974% of the 3.7 ms timed region (budget 20%) |
| neg_read_into_f64_n1048576 | pytorch-cuda | neg_read_into_f64_n1048576 untimed 958.6 ms is 25451% of the 3.8 ms timed region (budget 20%) |
| conj_into_c64_n1048576 | pytorch-cuda | conj_into_c64_n1048576 untimed 919.3 ms is 15172% of the 6.1 ms timed region (budget 20%) |
| conj_read_into_c64_n1048576 | pytorch-cuda | conj_read_into_c64_n1048576 untimed 904.5 ms is 14914% of the 6.1 ms timed region (budget 20%) |
| tensordot_f64_n262144 | pytorch-cuda | tensordot_f64_n262144 untimed 335.8 ms is 685% of the 49.0 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | pytorch-cuda | vdot_read_f64_n262144 untimed 11.1 ms is 31% of the 35.8 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | pytorch-cuda | norm_squared_read_f64_n262144 untimed 9.5 ms is 27% of the 34.7 ms timed region (budget 20%) |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | axpby_read_into_accum_f64_n1048576 untimed 903.3 ms is 11475% of the 7.9 ms timed region (budget 20%) |
| dot_general_read_into_f64_s256 | pytorch-cuda | dot_general_read_into_f64_s256 untimed 354.9 ms is 614% of the 57.8 ms timed region (budget 20%) |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | dot_general_read_into_accum_f64_s256 untimed 104.0 ms is 305% of the 34.1 ms timed region (budget 20%) |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | dot_general_read_into_accum_cached_f64_s256 untimed 88.5 ms is 259% of the 34.1 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | pytorch-cuda | add_strided_read_f64_n1048576 untimed 908.7 ms is 15290% of the 5.9 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | pytorch-cuda | mul_strided_read_f64_n1048576 untimed 913.4 ms is 15889% of the 5.7 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | pytorch-cuda | neg_strided_read_f64_n1048576 untimed 885.2 ms is 23929% of the 3.7 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | pytorch-cuda | exp_strided_read_f64_n1048576 untimed 979.2 ms is 25054% of the 3.9 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | tanh_strided_read_f64_n1048576 untimed 893.3 ms is 22873% of the 3.9 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_strided_read_f64_n1048576 untimed 27.3 ms is 53% of the 52.0 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | pytorch-cuda | conj_strided_read_c64_n1048576 untimed 938.3 ms is 15413% of the 6.1 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | pytorch-cuda | dot_general_batch_f64_s256 untimed 229.8 ms is 708% of the 32.5 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | dot_conj_l0r0_c64_n262144 untimed 15.8 ms is 33% of the 48.0 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | pytorch-cuda | einsum_plan_owned_f64_s256 untimed 78.6 ms is 200% of the 39.3 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | pytorch-cuda | einsum_plan_read_f64_s256 untimed 68.8 ms is 227% of the 30.3 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | einsum_plan_strided_read_f64_s256 untimed 66.3 ms is 218% of the 30.3 ms timed region (budget 20%) |
| einsum_plan_into_f64_s256 | pytorch-cuda | einsum_plan_into_f64_s256 untimed 69.6 ms is 122% of the 57.0 ms timed region (budget 20%) |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | einsum_plan_read_into_f64_s256 untimed 80.2 ms is 141% of the 56.7 ms timed region (budget 20%) |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_read_into_accum_f64_s256 untimed 88.6 ms is 262% of the 33.9 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_f64_s256 untimed 73.9 ms is 132% of the 56.1 ms timed region (budget 20%) |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | einsum_plan_strided_read_into_accum_f64_s256 untimed 100.0 ms is 298% of the 33.5 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | pytorch-cuda | sub_strided_read_f64_n1048576 untimed 903.5 ms is 15153% of the 6.0 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | pytorch-cuda | div_strided_read_f64_n1048576 untimed 909.1 ms is 14276% of the 6.4 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | pytorch-cuda | rem_strided_read_f64_n1048576 untimed 921.1 ms is 15406% of the 6.0 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | pytorch-cuda | pow_strided_read_f64_n1048576 untimed 922.8 ms is 6567% of the 14.1 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | maximum_strided_read_f64_n1048576 untimed 978.1 ms is 16262% of the 6.0 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | minimum_strided_read_f64_n1048576 untimed 924.4 ms is 15672% of the 5.9 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | pytorch-cuda | abs_strided_read_f64_n1048576 untimed 885.5 ms is 23771% of the 3.7 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | pytorch-cuda | sign_strided_read_f64_n1048576 untimed 1007.8 ms is 27661% of the 3.6 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | expm1_strided_read_f64_n1048576 untimed 884.0 ms is 18653% of the 4.7 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | pytorch-cuda | log_strided_read_f64_n1048576 untimed 965.8 ms is 16090% of the 6.0 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | log1p_strided_read_f64_n1048576 untimed 920.7 ms is 14774% of the 6.2 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | pytorch-cuda | erf_strided_read_f64_n1048576 untimed 898.7 ms is 13078% of the 6.9 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | pytorch-cuda | sin_strided_read_f64_n1048576 untimed 974.5 ms is 18587% of the 5.2 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | pytorch-cuda | cos_strided_read_f64_n1048576 untimed 891.4 ms is 16859% of the 5.3 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | sqrt_strided_read_f64_n1048576 untimed 883.0 ms is 23580% of the 3.7 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | rsqrt_strided_read_f64_n1048576 untimed 964.2 ms is 25596% of the 3.8 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | reduce_prod_strided_read_f64_n1048576 untimed 644.3 ms is 1243% of the 51.9 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | reduce_max_strided_read_f64_n1048576 untimed 19.7 ms is 38% of the 52.3 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | reduce_min_strided_read_f64_n1048576 untimed 20.2 ms is 38% of the 52.4 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | reduce_sum_squares_strided_read_f64_n1048576 untimed 19.9 ms is 49% of the 40.7 ms timed region (budget 20%) |
| tensordot_f64_n262144 | jax-cuda | tensordot_f64_n262144 untimed 843.3 ms is 1836% of the 45.9 ms timed region (budget 20%) |
| vdot_read_f64_n262144 | jax-cuda | vdot_read_f64_n262144 untimed 154.3 ms is 401% of the 38.5 ms timed region (budget 20%) |
| norm_squared_read_f64_n262144 | jax-cuda | norm_squared_read_f64_n262144 untimed 12.4 ms is 32% of the 38.3 ms timed region (budget 20%) |
| add_strided_read_f64_n1048576 | jax-cuda | add_strided_read_f64_n1048576 untimed 1063.5 ms is 4902% of the 21.7 ms timed region (budget 20%) |
| mul_strided_read_f64_n1048576 | jax-cuda | mul_strided_read_f64_n1048576 untimed 962.1 ms is 4723% of the 20.4 ms timed region (budget 20%) |
| neg_strided_read_f64_n1048576 | jax-cuda | neg_strided_read_f64_n1048576 untimed 940.7 ms is 5467% of the 17.2 ms timed region (budget 20%) |
| exp_strided_read_f64_n1048576 | jax-cuda | exp_strided_read_f64_n1048576 untimed 964.8 ms is 4405% of the 21.9 ms timed region (budget 20%) |
| tanh_strided_read_f64_n1048576 | jax-cuda | tanh_strided_read_f64_n1048576 untimed 970.0 ms is 4246% of the 22.8 ms timed region (budget 20%) |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | reduce_sum_strided_read_f64_n1048576 untimed 134.3 ms is 285% of the 47.2 ms timed region (budget 20%) |
| conj_strided_read_c64_n1048576 | jax-cuda | conj_strided_read_c64_n1048576 untimed 1060.4 ms is 6759% of the 15.7 ms timed region (budget 20%) |
| dot_general_batch_f64_s256 | jax-cuda | dot_general_batch_f64_s256 untimed 706.9 ms is 1570% of the 45.0 ms timed region (budget 20%) |
| dot_conj_l0r0_c64_n262144 | jax-cuda | dot_conj_l0r0_c64_n262144 untimed 121.5 ms is 347% of the 35.0 ms timed region (budget 20%) |
| einsum_plan_owned_f64_s256 | jax-cuda | einsum_plan_owned_f64_s256 untimed 448.8 ms is 1376% of the 32.6 ms timed region (budget 20%) |
| einsum_plan_read_f64_s256 | jax-cuda | einsum_plan_read_f64_s256 untimed 67.5 ms is 205% of the 33.0 ms timed region (budget 20%) |
| einsum_plan_strided_read_f64_s256 | jax-cuda | einsum_plan_strided_read_f64_s256 untimed 244.1 ms is 758% of the 32.2 ms timed region (budget 20%) |
| sub_strided_read_f64_n1048576 | jax-cuda | sub_strided_read_f64_n1048576 untimed 959.8 ms is 2724% of the 35.2 ms timed region (budget 20%) |
| div_strided_read_f64_n1048576 | jax-cuda | div_strided_read_f64_n1048576 untimed 957.6 ms is 2845% of the 33.7 ms timed region (budget 20%) |
| rem_strided_read_f64_n1048576 | jax-cuda | rem_strided_read_f64_n1048576 untimed 972.2 ms is 5826% of the 16.7 ms timed region (budget 20%) |
| pow_strided_read_f64_n1048576 | jax-cuda | pow_strided_read_f64_n1048576 untimed 982.0 ms is 5540% of the 17.7 ms timed region (budget 20%) |
| maximum_strided_read_f64_n1048576 | jax-cuda | maximum_strided_read_f64_n1048576 untimed 936.7 ms is 4731% of the 19.8 ms timed region (budget 20%) |
| minimum_strided_read_f64_n1048576 | jax-cuda | minimum_strided_read_f64_n1048576 untimed 962.4 ms is 4732% of the 20.3 ms timed region (budget 20%) |
| abs_strided_read_f64_n1048576 | jax-cuda | abs_strided_read_f64_n1048576 untimed 961.0 ms is 5977% of the 16.1 ms timed region (budget 20%) |
| sign_strided_read_f64_n1048576 | jax-cuda | sign_strided_read_f64_n1048576 untimed 941.3 ms is 5838% of the 16.1 ms timed region (budget 20%) |
| expm1_strided_read_f64_n1048576 | jax-cuda | expm1_strided_read_f64_n1048576 untimed 981.8 ms is 6010% of the 16.3 ms timed region (budget 20%) |
| log_strided_read_f64_n1048576 | jax-cuda | log_strided_read_f64_n1048576 untimed 952.8 ms is 5815% of the 16.4 ms timed region (budget 20%) |
| log1p_strided_read_f64_n1048576 | jax-cuda | log1p_strided_read_f64_n1048576 untimed 971.1 ms is 6078% of the 16.0 ms timed region (budget 20%) |
| erf_strided_read_f64_n1048576 | jax-cuda | erf_strided_read_f64_n1048576 untimed 976.8 ms is 4393% of the 22.2 ms timed region (budget 20%) |
| sin_strided_read_f64_n1048576 | jax-cuda | sin_strided_read_f64_n1048576 untimed 1019.9 ms is 6377% of the 16.0 ms timed region (budget 20%) |
| cos_strided_read_f64_n1048576 | jax-cuda | cos_strided_read_f64_n1048576 untimed 1008.5 ms is 5835% of the 17.3 ms timed region (budget 20%) |
| sqrt_strided_read_f64_n1048576 | jax-cuda | sqrt_strided_read_f64_n1048576 untimed 970.3 ms is 5905% of the 16.4 ms timed region (budget 20%) |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | rsqrt_strided_read_f64_n1048576 untimed 964.1 ms is 4038% of the 23.9 ms timed region (budget 20%) |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | reduce_prod_strided_read_f64_n1048576 untimed 134.7 ms is 372% of the 36.3 ms timed region (budget 20%) |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | reduce_max_strided_read_f64_n1048576 untimed 174.6 ms is 264% of the 66.2 ms timed region (budget 20%) |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | reduce_min_strided_read_f64_n1048576 untimed 75.8 ms is 223% of the 33.9 ms timed region (budget 20%) |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | reduce_sum_squares_strided_read_f64_n1048576 untimed 165.0 ms is 332% of the 49.6 ms timed region (budget 20%) |