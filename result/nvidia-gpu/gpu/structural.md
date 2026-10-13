# gpu/structural: CUDA public API benchmark

Measured cells: **534**. Unsupported cells: **56**, including **8** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/structural/20261013_211954/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

I32/I64 outputs require exact integer equality without conversion to floating point. Division and remainder truncate toward zero; axis-1 sum preserves the input dtype. Arithmetic fixtures avoid overflow, and I64 unary/order fixtures include values above 2**53 to detect loss of integer precision. A zero verification value in these rows means exact equality.

Reshape-copy cases add a leading singleton dimension and materialize the full output. Direct Tensor.reshape and EagerSession.reshape copy device data; PyTorch calls public view_copy. Direct/PyTorch initial/final allocation-pointer checks are untimed and reject input aliasing. Eager reshape copies its input before returning a descriptor view; original CUDA placement/shape/dtype are checked and a public inspection copy is made outside timing. JAX reshape has no established matching owned-output guarantee and remains unsupported; reshape-plus-copy composition is not substituted.

Stack selects actual public Tensor.stack/EagerSession.stack with two distinct inputs and axis 0 or 2. Their intrinsic intermediate allocations are part of this public stack operation. PyTorch/JAX use native stack; final outputs are fully verified. Direct Rust initial/final pointer checks reject aliasing either input outside timing; eager original descriptors are checked before any untimed inspection copy.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| pad_default_f32_n1024 | 11.107 (0.700) | 87.698 (4.400) | 27.356 (0.272) | 113.978 (1.163) | 85.602 (17.079) |
| pad_default_f64_n1024 | 11.305 (1.194) | 100.325 (5.146) | 26.899 (0.655) | 136.479 (15.113) | 83.532 (16.123) |
| slice_default_f32_n1024 | 14.435 (0.813) | unsupported | 21.947 (0.128) | 139.657 (12.717) | 83.622 (11.329) |
| slice_default_f64_n1024 | 16.119 (1.674) | unsupported | 22.539 (0.465) | 140.175 (12.970) | 80.144 (14.563) |
| reverse_default_f32_n1024 | 17.368 (0.574) | 81.351 (22.484) | 12.324 (0.206) | 97.026 (19.510) | 79.852 (10.068) |
| reverse_default_f64_n1024 | 17.181 (0.370) | 103.598 (9.140) | 12.889 (0.277) | 94.402 (16.507) | 74.335 (1.194) |
| concatenate_default_f32_n1024 | 27.092 (3.417) | 227.647 (6.927) | 11.834 (0.779) | 160.046 (9.261) | 77.459 (0.461) |
| concatenate_default_f64_n1024 | 28.917 (1.022) | 222.489 (5.662) | 11.858 (0.102) | 155.635 (3.938) | 79.805 (4.057) |
| extract_diag_default_f32_n1024 | 16.941 (0.182) | 136.703 (4.080) | 16.466 (0.103) | 82.632 (8.864) | 74.809 (8.482) |
| extract_diag_default_f64_n1024 | 17.045 (0.327) | 139.436 (3.543) | 16.276 (0.111) | 81.981 (11.373) | 77.538 (3.717) |
| embed_diag_default_f32_n32 | 25.936 (0.456) | 171.018 (4.202) | 23.201 (1.168) | 86.017 (10.403) | 77.756 (8.320) |
| embed_diag_default_f64_n32 | 25.681 (1.662) | 180.611 (5.400) | 22.038 (0.458) | 88.906 (12.144) | 70.026 (3.453) |
| tril_default_f32_n1024 | 16.278 (1.005) | 166.580 (5.911) | 7.789 (0.061) | 80.449 (7.249) | 75.091 (10.980) |
| tril_default_f64_n1024 | 16.419 (0.684) | 167.920 (40.213) | 7.814 (0.075) | 79.017 (7.123) | 80.978 (7.880) |
| triu_default_f32_n1024 | 16.496 (0.310) | 141.788 (3.787) | 7.758 (0.069) | 79.266 (7.582) | 75.980 (12.665) |
| triu_default_f64_n1024 | 16.270 (0.273) | 145.285 (6.638) | 7.787 (0.129) | 79.948 (7.122) | 80.507 (11.379) |
| take_along_axis_broadcast_indices_f32_n1024 | 86.799 (6.433) | 132.796 (4.973) | 37.745 (0.852) | 82.327 (8.477) | 83.358 (8.032) |
| take_along_axis_batched_indices_f32_n1024 | unsupported | unsupported | 39.098 (1.655) | 83.040 (8.325) | 83.845 (8.491) |
| take_along_axis_broadcast_indices_f64_n1024 | 132.385 (7.163) | 145.514 (4.400) | 40.226 (2.605) | 82.818 (7.751) | 83.089 (10.825) |
| take_along_axis_batched_indices_f64_n1024 | unsupported | unsupported | 36.999 (3.546) | 86.342 (13.233) | 85.600 (13.085) |
| scatter_default_f32_n1024 | 41.029 (0.510) | 499.825 (22.273) | 24.599 (0.526) | 180.736 (20.408) | 95.401 (14.711) |
| scatter_default_f64_n1024 | 40.449 (0.336) | 532.206 (40.486) | 24.876 (0.765) | 168.609 (25.336) | 92.036 (11.900) |
| cast_default_f32_n1024 | 14.224 (1.755) | 211.408 (33.545) | 12.497 (0.190) | 141.552 (11.362) | 76.817 (8.075) |
| cast_default_f64_n1024 | 12.962 (1.758) | 155.781 (3.016) | 12.348 (0.124) | 142.158 (13.551) | 75.048 (4.367) |
| pad_default_f32_n1048576 | 70.836 (0.191) | 90.879 (4.316) | 42.563 (0.104) | 115.978 (9.868) | 73.082 (3.042) |
| pad_default_f64_n1048576 | 70.547 (0.274) | 120.674 (0.170) | 80.551 (0.078) | 115.556 (3.872) | 67.741 (2.722) |
| slice_default_f32_n1048576 | 17.588 (0.047) | unsupported | 23.492 (0.639) | 109.296 (2.317) | 72.472 (2.600) |
| slice_default_f64_n1048576 | 30.577 (0.156) | unsupported | 28.531 (0.045) | 106.393 (3.187) | 73.362 (3.638) |
| reverse_default_f32_n1048576 | 32.886 (0.044) | 60.893 (0.704) | 28.865 (0.027) | 79.390 (1.700) | 71.700 (1.945) |
| reverse_default_f64_n1048576 | 58.184 (0.095) | 105.449 (0.059) | 54.414 (0.110) | 79.093 (2.913) | 67.639 (1.578) |
| concatenate_default_f32_n1048576 | 67.011 (0.185) | 118.345 (0.333) | 54.202 (0.062) | 142.983 (2.040) | 69.586 (2.033) |
| concatenate_default_f64_n1048576 | 118.458 (1.231) | 215.604 (0.163) | 105.518 (0.122) | 148.146 (4.359) | 114.814 (3.201) |
| extract_diag_default_f32_n1048576 | 11.559 (0.441) | 58.310 (1.534) | 16.709 (0.195) | 75.258 (2.328) | 72.666 (4.447) |
| extract_diag_default_f64_n1048576 | 10.722 (0.368) | 58.413 (0.795) | 16.812 (0.241) | 83.918 (4.085) | 75.180 (2.295) |
| embed_diag_default_f32_n1024 | 22.002 (0.836) | 141.458 (11.995) | 24.173 (1.388) | 80.358 (3.416) | 71.800 (3.206) |
| embed_diag_default_f64_n1024 | 32.243 (0.587) | 160.651 (20.266) | 29.803 (0.134) | 78.215 (3.274) | 73.263 (2.259) |
| tril_default_f32_n1048576 | 28.810 (0.216) | 54.459 (0.074) | 22.724 (0.169) | 71.009 (1.849) | 71.130 (1.408) |
| tril_default_f64_n1048576 | 44.662 (0.679) | 95.657 (0.378) | 42.077 (0.130) | 68.988 (2.201) | 68.958 (1.302) |
| triu_default_f32_n1048576 | 29.139 (0.207) | 54.834 (0.211) | 22.545 (0.178) | 69.746 (1.175) | 73.998 (1.494) |
| triu_default_f64_n1048576 | 45.242 (0.466) | 95.759 (0.119) | 41.871 (0.045) | 65.257 (0.854) | 72.178 (5.445) |
| take_along_axis_broadcast_indices_f32_n1048576 | 114.405 (7.117) | 133.644 (9.793) | 50.382 (0.138) | 81.321 (9.025) | 83.124 (7.056) |
| take_along_axis_batched_indices_f32_n1048576 | unsupported | unsupported | 63.414 (0.307) | 81.386 (9.977) | 76.164 (3.922) |
| take_along_axis_broadcast_indices_f64_n1048576 | 181.732 (5.164) | 116.547 (10.279) | 69.891 (0.457) | 83.476 (11.299) | 83.872 (10.383) |
| take_along_axis_batched_indices_f64_n1048576 | unsupported | unsupported | 81.990 (0.136) | 82.523 (5.544) | 84.219 (8.627) |
| scatter_default_f32_n1048576 | 80.306 (1.071) | 149.651 (1.587) | 74.855 (0.095) | 159.374 (8.875) | 83.739 (2.762) |
| scatter_default_f64_n1048576 | 143.782 (1.890) | 244.063 (1.738) | 132.218 (0.145) | 164.239 (5.849) | 142.379 (2.581) |
| cast_default_f32_n1048576 | 44.004 (0.365) | 69.432 (0.225) | 41.258 (0.059) | 128.776 (6.314) | 68.280 (3.248) |
| cast_default_f64_n1048576 | 41.893 (0.079) | 94.692 (0.112) | 40.444 (0.030) | 124.862 (4.155) | 67.097 (3.579) |
| broadcast_in_dim_f32_n1024 | 10.774 (0.992) | unsupported | 17.331 (1.088) | 101.375 (4.293) | 88.738 (17.717) |
| broadcast_in_dim_f64_n1024 | 11.975 (1.330) | unsupported | 16.508 (0.563) | 101.695 (8.271) | 85.791 (8.034) |
| convert_f32_f64_n1024 | 8.519 (0.648) | 157.975 (21.569) | 12.673 (0.192) | 119.369 (4.102) | 82.230 (2.641) |
| broadcast_in_dim_f32_n1048576 | 30.885 (0.519) | unsupported | 17.510 (0.961) | 105.774 (1.791) | 78.759 (1.429) |
| broadcast_in_dim_f64_n1048576 | 33.040 (0.636) | unsupported | 28.095 (0.073) | 104.703 (1.331) | 80.337 (1.254) |
| convert_f32_f64_n1048576 | 44.324 (0.589) | 69.609 (0.141) | 41.323 (0.050) | 125.788 (2.808) | 78.319 (2.312) |
| dynamic_slice_in_bounds_f32_n1024 | 83.709 (2.325) | 356.412 (143.022) | unsupported | 150.465 (17.842) | 89.751 (15.040) |
| dynamic_slice_clamped_f32_n1024 | 83.044 (0.996) | 570.227 (122.261) | unsupported | 127.366 (6.479) | 89.759 (14.166) |
| gather_axis0_f32_n1024 | 27.344 (0.281) | 33.652 (0.412) | 11.730 (0.226) | 123.493 (2.151) | 88.106 (10.768) |
| gather_axis1_f32_n1024 | 23.094 (2.130) | 33.682 (1.258) | 11.311 (0.076) | 134.902 (8.230) | 87.415 (8.377) |
| dynamic_slice_in_bounds_f64_n1024 | 24.514 (2.203) | 488.001 (10.741) | unsupported | 155.368 (13.658) | 87.604 (9.097) |
| dynamic_slice_clamped_f64_n1024 | 19.329 (1.713) | 570.749 (42.849) | unsupported | 129.460 (7.061) | 89.274 (11.768) |
| gather_axis0_f64_n1024 | 23.969 (3.057) | 34.490 (2.260) | 11.532 (0.094) | 127.731 (3.061) | 86.189 (11.062) |
| gather_axis1_f64_n1024 | 20.370 (0.853) | 37.579 (1.612) | 11.768 (0.656) | 137.221 (10.186) | 87.109 (8.981) |
| dynamic_slice_in_bounds_f32_n1048576 | 16.239 (0.396) | 573.338 (47.360) | unsupported | 154.928 (23.469) | 89.485 (13.195) |
| dynamic_slice_clamped_f32_n1048576 | 15.376 (0.342) | 799.882 (35.694) | unsupported | 126.036 (4.115) | 84.191 (8.147) |
| gather_axis0_f32_n1048576 | 23.737 (0.056) | 43.037 (2.300) | 22.674 (0.526) | 115.543 (3.179) | 83.530 (1.934) |
| gather_axis1_f32_n1048576 | 20.457 (0.040) | 30.863 (1.311) | 22.779 (0.419) | 119.706 (3.083) | 83.691 (4.128) |
| dynamic_slice_in_bounds_f64_n1048576 | 17.096 (0.069) | 721.551 (48.368) | unsupported | 137.560 (4.973) | 84.558 (7.359) |
| dynamic_slice_clamped_f64_n1048576 | 16.960 (0.061) | 730.490 (59.736) | unsupported | 129.294 (0.703) | 83.946 (10.273) |
| gather_axis0_f64_n1048576 | 44.160 (0.072) | 47.092 (0.462) | 28.150 (0.054) | 115.036 (3.714) | 83.276 (7.243) |
| gather_axis1_f64_n1048576 | 32.598 (0.150) | 34.541 (0.169) | 41.749 (0.070) | 128.899 (5.634) | 82.117 (9.708) |
| pad_interior_f32_n1024 | 85.783 (2.585) | 523.427 (34.167) | unsupported | 107.427 (15.241) | 84.367 (4.244) |
| pad_crop_f32_n1024 | 81.629 (1.708) | 568.739 (38.520) | 30.030 (2.076) | 113.559 (7.248) | 86.882 (12.784) |
| pad_interior_crop_f32_n1024 | 75.047 (0.729) | 492.161 (29.027) | unsupported | 116.458 (9.192) | 90.739 (11.141) |
| pad_interior_f64_n1024 | 75.537 (0.703) | 328.809 (19.290) | unsupported | 135.810 (7.043) | 96.806 (22.561) |
| pad_crop_f64_n1024 | 78.967 (0.316) | 379.009 (17.360) | 29.802 (0.724) | 149.378 (14.125) | 101.672 (9.778) |
| pad_interior_crop_f64_n1024 | 77.301 (1.044) | 334.399 (21.644) | unsupported | 170.729 (31.740) | 100.724 (16.686) |
| pad_interior_f32_n1048576 | 323.387 (0.980) | 370.760 (0.791) | unsupported | 116.385 (1.648) | 103.515 (1.356) |
| pad_crop_f32_n1048576 | 59.886 (0.211) | 91.373 (0.056) | 42.097 (0.045) | 108.041 (2.580) | 80.138 (2.355) |
| pad_interior_crop_f32_n1048576 | 310.774 (1.425) | 353.100 (0.246) | unsupported | 113.092 (1.596) | 104.111 (3.013) |
| pad_interior_f64_n1048576 | 332.206 (1.080) | 395.625 (0.700) | unsupported | 219.326 (8.047) | 207.460 (6.011) |
| pad_crop_f64_n1048576 | 66.147 (0.281) | 117.200 (0.166) | 79.829 (0.088) | 107.863 (2.761) | 73.397 (2.034) |
| pad_interior_crop_f64_n1048576 | 330.143 (1.335) | 378.307 (0.294) | unsupported | 214.948 (18.323) | 214.250 (10.283) |
| reshape_copy_f32_n1024 | 74.182 (0.777) | 127.976 (6.093) | 17.798 (0.307) | unsupported | unsupported |
| stack_axis0_f32_n1024 | 253.274 (19.195) | 774.774 (118.705) | 14.415 (0.160) | 139.181 (7.196) | 84.999 (2.939) |
| stack_axis2_f32_n1024 | 331.720 (7.143) | 838.139 (17.170) | 14.675 (0.395) | 145.819 (5.338) | 81.522 (2.696) |
| reshape_copy_f64_n1024 | 97.782 (0.955) | 149.051 (4.615) | 19.324 (0.927) | unsupported | unsupported |
| stack_axis0_f64_n1024 | 408.776 (90.519) | 690.391 (30.310) | 15.273 (0.801) | 155.673 (23.383) | 85.440 (7.103) |
| stack_axis2_f64_n1024 | 412.245 (13.439) | 840.486 (24.116) | 15.149 (0.615) | 156.105 (24.636) | 87.120 (6.818) |
| reshape_copy_c32_n1024 | 110.743 (1.015) | 146.292 (4.555) | 17.827 (0.130) | unsupported | unsupported |
| stack_axis0_c32_n1024 | 446.477 (43.358) | 848.945 (43.654) | 14.084 (0.208) | 161.391 (26.988) | 87.309 (5.220) |
| stack_axis2_c32_n1024 | 420.079 (28.658) | 885.885 (63.068) | 14.231 (0.259) | 157.787 (10.911) | 88.034 (9.218) |
| reshape_copy_c64_n1024 | 105.933 (2.937) | 136.529 (6.374) | 18.658 (1.379) | unsupported | unsupported |
| stack_axis0_c64_n1024 | 406.911 (6.852) | 706.783 (12.151) | 13.992 (0.433) | 161.270 (19.081) | 83.341 (1.377) |
| stack_axis2_c64_n1024 | 420.004 (21.919) | 722.171 (22.283) | 14.292 (0.199) | 159.539 (23.127) | 87.124 (5.840) |
| reshape_copy_i32_n1024 | 95.802 (1.490) | 133.007 (1.514) | 17.815 (0.427) | unsupported | unsupported |
| stack_axis0_i32_n1024 | 281.356 (83.498) | 502.351 (7.357) | 14.537 (0.762) | 164.245 (30.552) | 88.188 (7.333) |
| stack_axis2_i32_n1024 | 332.110 (36.139) | 586.126 (6.802) | 14.664 (0.134) | 147.382 (0.383) | 86.778 (7.873) |
| reshape_copy_i64_n1024 | 78.067 (1.026) | 138.258 (2.276) | 18.097 (0.258) | unsupported | unsupported |
| stack_axis0_i64_n1024 | 308.625 (3.861) | 934.308 (16.614) | 14.515 (0.741) | 137.101 (1.929) | 87.408 (6.939) |
| stack_axis2_i64_n1024 | 486.923 (12.169) | 851.653 (15.847) | 14.384 (0.409) | 136.084 (7.961) | 83.454 (7.990) |
| reshape_copy_f32_n1048576 | 35.599 (1.391) | 55.274 (7.121) | 27.501 (0.033) | unsupported | unsupported |
| stack_axis0_f32_n1048576 | 145.942 (0.699) | 206.345 (1.306) | 53.814 (0.070) | 140.087 (3.180) | 78.462 (1.511) |
| stack_axis2_f32_n1048576 | 135.528 (4.458) | 199.473 (2.281) | 88.609 (0.141) | 142.063 (3.336) | 76.046 (1.194) |
| reshape_copy_f64_n1048576 | 55.008 (0.081) | 55.174 (0.119) | 53.232 (0.046) | unsupported | unsupported |
| stack_axis0_f64_n1048576 | 281.916 (0.176) | 390.573 (0.270) | 105.142 (0.102) | 143.838 (2.001) | 110.387 (0.667) |
| stack_axis2_f64_n1048576 | 215.449 (0.149) | 326.493 (0.256) | 173.731 (0.248) | 141.331 (2.832) | 110.238 (0.462) |
| reshape_copy_c32_n1048576 | 55.353 (0.231) | 55.306 (0.097) | 53.240 (0.089) | unsupported | unsupported |
| stack_axis0_c32_n1048576 | 281.925 (0.446) | 391.568 (1.616) | 105.330 (0.167) | 155.206 (2.826) | 110.383 (0.202) |
| stack_axis2_c32_n1048576 | 216.185 (0.350) | 327.642 (1.784) | 173.636 (0.124) | 148.334 (4.694) | 110.365 (0.400) |
| reshape_copy_c64_n1048576 | 106.562 (0.524) | 106.680 (0.262) | 104.398 (0.074) | unsupported | unsupported |
| stack_axis0_c64_n1048576 | 553.458 (0.621) | 763.959 (0.888) | 207.647 (0.282) | 252.044 (19.983) | 220.342 (2.085) |
| stack_axis2_c64_n1048576 | 419.412 (0.948) | 633.826 (2.013) | 347.662 (0.436) | 243.087 (11.131) | 222.701 (6.456) |
| reshape_copy_i32_n1048576 | 30.668 (0.224) | 31.109 (0.225) | 27.604 (0.067) | unsupported | unsupported |
| stack_axis0_i32_n1048576 | 154.736 (1.250) | 209.551 (1.237) | 53.873 (0.108) | 147.547 (10.553) | 77.631 (2.730) |
| stack_axis2_i32_n1048576 | 143.266 (0.672) | 194.409 (2.049) | 89.622 (0.101) | 155.021 (17.594) | 80.162 (5.279) |
| reshape_copy_i64_n1048576 | 58.738 (0.532) | 58.990 (0.438) | 53.109 (0.197) | unsupported | unsupported |
| stack_axis0_i64_n1048576 | 299.986 (2.713) | 404.012 (2.212) | 105.052 (0.164) | 169.309 (17.506) | 111.155 (0.210) |
| stack_axis2_i64_n1048576 | 224.293 (1.488) | 340.523 (1.545) | 173.558 (0.082) | 165.199 (14.024) | 111.428 (0.601) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| pad_default_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.843 | 0 |
| pad_default_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.894 | 0 |
| slice_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 1.848 | 0 |
| slice_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.063 | 0 |
| reverse_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.223 | 0 |
| reverse_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.199 | 0 |
| concatenate_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.468 | 0 |
| concatenate_default_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.851 | 0 |
| extract_diag_default_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.084 | 0 |
| extract_diag_default_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.091 | 0 |
| embed_diag_default_f32_n32 | tenferro-cuda-direct | 128 | 15 | 3.320 | 0 |
| embed_diag_default_f64_n32 | tenferro-cuda-direct | 128 | 15 | 3.287 | 0 |
| tril_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.084 | 0 |
| tril_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.102 | 0 |
| triu_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.111 | 0 |
| triu_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.082 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.778 | 0 |
| take_along_axis_batched_indices_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.118 | 0 |
| take_along_axis_batched_indices_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.626 | 0 |
| scatter_default_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.589 | 0 |
| cast_default_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.641 | 0 |
| cast_default_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.318 | 0 |
| pad_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.267 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-direct | 31 | 15 | 2.187 | 0 |
| slice_default_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.251 | 0 |
| slice_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.957 | 0 |
| reverse_default_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.105 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.862 | 0 |
| concatenate_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.144 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.895 | 0 |
| extract_diag_default_f32_n1048576 | tenferro-cuda-direct | 256 | 15 | 2.959 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-direct | 256 | 15 | 2.745 | 0 |
| embed_diag_default_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.408 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 1.032 | 0 |
| tril_default_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.844 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.429 | 0 |
| triu_default_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.865 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.448 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.661 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.908 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.570 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.301 | 0 |
| cast_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.408 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.681 | 0 |
| broadcast_in_dim_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.758 | 0 |
| broadcast_in_dim_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.066 | 0 |
| convert_f32_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.181 | 0 |
| broadcast_in_dim_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.977 | 0 |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.057 | 0 |
| convert_f32_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.418 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.679 | 0 |
| dynamic_slice_clamped_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.657 | 0 |
| gather_axis0_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.500 | 0 |
| gather_axis1_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.956 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.138 | 0 |
| dynamic_slice_clamped_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.474 | 0 |
| gather_axis0_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.068 | 0 |
| gather_axis1_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.607 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.079 | 0 |
| dynamic_slice_clamped_f32_n1048576 | tenferro-cuda-direct | 256 | 15 | 3.936 | 0 |
| gather_axis0_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 3.038 | 0 |
| gather_axis1_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.618 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.188 | 0 |
| dynamic_slice_clamped_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.171 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.826 | 0 |
| gather_axis1_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.086 | 0 |
| pad_interior_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.745 | 0 |
| pad_crop_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.612 | 0 |
| pad_interior_crop_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.402 | 0 |
| pad_interior_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.417 | 0 |
| pad_crop_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.527 | 0 |
| pad_interior_crop_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.474 | 0 |
| pad_interior_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.587 | 0 |
| pad_crop_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.833 | 0 |
| pad_interior_crop_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.486 | 0 |
| pad_interior_f64_n1048576 | tenferro-cuda-direct | 5 | 15 | 1.661 | 0 |
| pad_crop_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.117 | 0 |
| pad_interior_crop_f64_n1048576 | tenferro-cuda-direct | 5 | 15 | 1.651 | 0 |
| reshape_copy_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.374 | 0 |
| stack_axis0_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 4.052 | 0 |
| stack_axis2_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.654 | 0 |
| reshape_copy_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 3.129 | 0 |
| stack_axis0_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 3.270 | 0 |
| stack_axis2_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 3.298 | 0 |
| reshape_copy_c32_n1024 | tenferro-cuda-direct | 32 | 15 | 3.544 | 0 |
| stack_axis0_c32_n1024 | tenferro-cuda-direct | 8 | 15 | 3.572 | 0 |
| stack_axis2_c32_n1024 | tenferro-cuda-direct | 8 | 15 | 3.361 | 0 |
| reshape_copy_c64_n1024 | tenferro-cuda-direct | 32 | 15 | 3.390 | 0 |
| stack_axis0_c64_n1024 | tenferro-cuda-direct | 8 | 15 | 3.255 | 0 |
| stack_axis2_c64_n1024 | tenferro-cuda-direct | 8 | 15 | 3.360 | 0 |
| reshape_copy_i32_n1024 | tenferro-cuda-direct | 32 | 15 | 3.066 | 0 |
| stack_axis0_i32_n1024 | tenferro-cuda-direct | 16 | 15 | 4.502 | 0 |
| stack_axis2_i32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.657 | 0 |
| reshape_copy_i64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.498 | 0 |
| stack_axis0_i64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.469 | 0 |
| stack_axis2_i64_n1024 | tenferro-cuda-direct | 8 | 15 | 3.895 | 0 |
| reshape_copy_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.278 | 0 |
| stack_axis0_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.335 | 0 |
| stack_axis2_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.168 | 0 |
| reshape_copy_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.760 | 0 |
| stack_axis0_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.255 | 0 |
| stack_axis2_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.447 | 0 |
| reshape_copy_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.771 | 0 |
| stack_axis0_c32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.255 | 0 |
| stack_axis2_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.459 | 0 |
| reshape_copy_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.705 | 0 |
| stack_axis0_c64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.214 | 0 |
| stack_axis2_c64_n1048576 | tenferro-cuda-direct | 8 | 15 | 3.355 | 0 |
| reshape_copy_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.963 | 0 |
| stack_axis0_i32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.476 | 0 |
| stack_axis2_i32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.292 | 0 |
| reshape_copy_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.880 | 0 |
| stack_axis0_i64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.400 | 0 |
| stack_axis2_i64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.589 | 0 |
| pad_default_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.806 | 0 |
| pad_default_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 3.210 | 0 |
| slice_default_f32_n1024 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| slice_default_f64_n1024 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| reverse_default_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.603 | 0 |
| reverse_default_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 3.315 | 0 |
| concatenate_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.642 | 0 |
| concatenate_default_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 1.780 | 0 |
| extract_diag_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.187 | 0 |
| extract_diag_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.231 | 0 |
| embed_diag_default_f32_n32 | tenferro-cuda-eager | 16 | 15 | 2.736 | 0 |
| embed_diag_default_f64_n32 | tenferro-cuda-eager | 16 | 15 | 2.890 | 0 |
| tril_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.665 | 0 |
| tril_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.687 | 0 |
| triu_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.269 | 0 |
| triu_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.325 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.125 | 0 |
| take_along_axis_batched_indices_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.328 | 0 |
| take_along_axis_batched_indices_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.999 | 0 |
| scatter_default_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.129 | 0 |
| cast_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.383 | 0 |
| cast_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.492 | 0 |
| pad_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.908 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-eager | 31 | 15 | 3.741 | 0 |
| slice_default_f32_n1048576 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| slice_default_f64_n1048576 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| reverse_default_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.897 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.374 | 0 |
| concatenate_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.787 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.450 | 0 |
| extract_diag_default_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.732 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.738 | 0 |
| embed_diag_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.263 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.570 | 0 |
| tril_default_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.485 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.061 | 0 |
| triu_default_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.509 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.064 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.138 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.729 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.394 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.905 | 0 |
| cast_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.222 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.030 | 0 |
| broadcast_in_dim_f32_n1024 | tenferro-cuda-eager | — | — | — | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| broadcast_in_dim_f64_n1024 | tenferro-cuda-eager | — | — | — | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| convert_f32_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.528 | 0 |
| broadcast_in_dim_f32_n1048576 | tenferro-cuda-eager | — | — | — | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-eager | — | — | — | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| convert_f32_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.227 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.851 | 0 |
| dynamic_slice_clamped_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 4.562 | 0 |
| gather_axis0_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.154 | 0 |
| gather_axis1_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.156 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.904 | 0 |
| dynamic_slice_clamped_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 4.566 | 0 |
| gather_axis0_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.207 | 0 |
| gather_axis1_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.405 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.293 | 0 |
| dynamic_slice_clamped_f32_n1048576 | tenferro-cuda-eager | 4 | 15 | 3.200 | 0 |
| gather_axis0_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.754 | 0 |
| gather_axis1_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.975 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.886 | 0 |
| dynamic_slice_clamped_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.922 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.014 | 0 |
| gather_axis1_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.211 | 0 |
| pad_interior_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.094 | 0 |
| pad_crop_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 4.550 | 0 |
| pad_interior_crop_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 1.969 | 0 |
| pad_interior_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.630 | 0 |
| pad_crop_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.032 | 0 |
| pad_interior_crop_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.675 | 0 |
| pad_interior_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.966 | 0 |
| pad_crop_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.924 | 0 |
| pad_interior_crop_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.825 | 0 |
| pad_interior_f64_n1048576 | tenferro-cuda-eager | 5 | 15 | 1.978 | 0 |
| pad_crop_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.750 | 0 |
| pad_interior_crop_f64_n1048576 | tenferro-cuda-eager | 5 | 15 | 1.892 | 0 |
| reshape_copy_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 4.095 | 0 |
| stack_axis0_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 3.099 | 0 |
| stack_axis2_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 3.353 | 0 |
| reshape_copy_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.385 | 0 |
| stack_axis0_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.762 | 0 |
| stack_axis2_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 3.362 | 0 |
| reshape_copy_c32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.341 | 0 |
| stack_axis0_c32_n1024 | tenferro-cuda-eager | 4 | 15 | 3.396 | 0 |
| stack_axis2_c32_n1024 | tenferro-cuda-eager | 4 | 15 | 3.544 | 0 |
| reshape_copy_c64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.184 | 0 |
| stack_axis0_c64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.827 | 0 |
| stack_axis2_c64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.889 | 0 |
| reshape_copy_i32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.128 | 0 |
| stack_axis0_i32_n1024 | tenferro-cuda-eager | 8 | 15 | 4.019 | 0 |
| stack_axis2_i32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.345 | 0 |
| reshape_copy_i64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.212 | 0 |
| stack_axis0_i64_n1024 | tenferro-cuda-eager | 2 | 15 | 1.869 | 0 |
| stack_axis2_i64_n1024 | tenferro-cuda-eager | 4 | 15 | 3.407 | 0 |
| reshape_copy_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.538 | 0 |
| stack_axis0_f32_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.302 | 0 |
| stack_axis2_f32_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.192 | 0 |
| reshape_copy_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.766 | 0 |
| stack_axis0_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 3.125 | 0 |
| stack_axis2_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.612 | 0 |
| reshape_copy_c32_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.770 | 0 |
| stack_axis0_c32_n1048576 | tenferro-cuda-eager | 8 | 15 | 3.133 | 0 |
| stack_axis2_c32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.621 | 0 |
| reshape_copy_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.707 | 0 |
| stack_axis0_c64_n1048576 | tenferro-cuda-eager | 4 | 15 | 3.056 | 0 |
| stack_axis2_c64_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.535 | 0 |
| reshape_copy_i32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.991 | 0 |
| stack_axis0_i32_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.353 | 0 |
| stack_axis2_i32_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.111 | 0 |
| reshape_copy_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.888 | 0 |
| stack_axis0_i64_n1048576 | tenferro-cuda-eager | 8 | 15 | 3.232 | 0 |
| stack_axis2_i64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.724 | 0 |
| pad_default_f32_n1024 | pytorch-cuda | 128 | 15 | 3.502 | 0 |
| pad_default_f64_n1024 | pytorch-cuda | 128 | 15 | 3.443 | 0 |
| slice_default_f32_n1024 | pytorch-cuda | 128 | 15 | 2.809 | 0 |
| slice_default_f64_n1024 | pytorch-cuda | 128 | 15 | 2.885 | 0 |
| reverse_default_f32_n1024 | pytorch-cuda | 256 | 15 | 3.155 | 0 |
| reverse_default_f64_n1024 | pytorch-cuda | 256 | 15 | 3.300 | 0 |
| concatenate_default_f32_n1024 | pytorch-cuda | 256 | 15 | 3.030 | 0 |
| concatenate_default_f64_n1024 | pytorch-cuda | 256 | 15 | 3.036 | 0 |
| extract_diag_default_f32_n1024 | pytorch-cuda | 128 | 15 | 2.108 | 0 |
| extract_diag_default_f64_n1024 | pytorch-cuda | 128 | 15 | 2.083 | 0 |
| embed_diag_default_f32_n32 | pytorch-cuda | 128 | 15 | 2.970 | 0 |
| embed_diag_default_f64_n32 | pytorch-cuda | 128 | 15 | 2.821 | 0 |
| tril_default_f32_n1024 | pytorch-cuda | 256 | 15 | 1.994 | 0 |
| tril_default_f64_n1024 | pytorch-cuda | 256 | 15 | 2.000 | 0 |
| triu_default_f32_n1024 | pytorch-cuda | 512 | 15 | 3.972 | 0 |
| triu_default_f64_n1024 | pytorch-cuda | 256 | 15 | 1.993 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | pytorch-cuda | 64 | 15 | 2.416 | 0 |
| take_along_axis_batched_indices_f32_n1024 | pytorch-cuda | 64 | 15 | 2.502 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | pytorch-cuda | 64 | 15 | 2.574 | 0 |
| take_along_axis_batched_indices_f64_n1024 | pytorch-cuda | 64 | 15 | 2.368 | 0 |
| scatter_default_f32_n1024 | pytorch-cuda | 128 | 15 | 3.149 | 0 |
| scatter_default_f64_n1024 | pytorch-cuda | 128 | 15 | 3.184 | 0 |
| cast_default_f32_n1024 | pytorch-cuda | 256 | 15 | 3.199 | 0 |
| cast_default_f64_n1024 | pytorch-cuda | 256 | 15 | 3.161 | 0 |
| pad_default_f32_n1048576 | pytorch-cuda | 63 | 15 | 2.681 | 0 |
| pad_default_f64_n1048576 | pytorch-cuda | 31 | 15 | 2.497 | 0 |
| slice_default_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.007 | 0 |
| slice_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 1.826 | 0 |
| reverse_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.847 | 0 |
| reverse_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.741 | 0 |
| concatenate_default_f32_n1048576 | pytorch-cuda | 32 | 15 | 1.734 | 0 |
| concatenate_default_f64_n1048576 | pytorch-cuda | 16 | 15 | 1.688 | 0 |
| extract_diag_default_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.139 | 0 |
| extract_diag_default_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.152 | 0 |
| embed_diag_default_f32_n1024 | pytorch-cuda | 64 | 15 | 1.547 | 0 |
| embed_diag_default_f64_n1024 | pytorch-cuda | 32 | 15 | 0.954 | 0 |
| tril_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.454 | 0 |
| tril_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.346 | 0 |
| triu_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.443 | 0 |
| triu_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.340 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | pytorch-cuda | 64 | 15 | 3.224 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | pytorch-cuda | 32 | 15 | 2.029 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.237 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.624 | 0 |
| scatter_default_f32_n1048576 | pytorch-cuda | 32 | 15 | 2.395 | 0 |
| scatter_default_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.115 | 0 |
| cast_default_f32_n1048576 | pytorch-cuda | 32 | 15 | 1.320 | 0 |
| cast_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.588 | 0 |
| broadcast_in_dim_f32_n1024 | pytorch-cuda | 128 | 15 | 2.218 | 0 |
| broadcast_in_dim_f64_n1024 | pytorch-cuda | 128 | 15 | 2.113 | 0 |
| convert_f32_f64_n1024 | pytorch-cuda | 256 | 15 | 3.244 | 0 |
| broadcast_in_dim_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.121 | 0 |
| broadcast_in_dim_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.899 | 0 |
| convert_f32_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.322 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_clamped_f32_n1024 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| gather_axis0_f32_n1024 | pytorch-cuda | 256 | 15 | 3.003 | 0 |
| gather_axis1_f32_n1024 | pytorch-cuda | 256 | 15 | 2.896 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_clamped_f64_n1024 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| gather_axis0_f64_n1024 | pytorch-cuda | 256 | 15 | 2.952 | 0 |
| gather_axis1_f64_n1024 | pytorch-cuda | 256 | 15 | 3.013 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_clamped_f32_n1048576 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| gather_axis0_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.902 | 0 |
| gather_axis1_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.916 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_clamped_f64_n1048576 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| gather_axis0_f64_n1048576 | pytorch-cuda | 64 | 15 | 1.802 | 0 |
| gather_axis1_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.672 | 0 |
| pad_interior_f32_n1024 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_crop_f32_n1024 | pytorch-cuda | 128 | 15 | 3.844 | 0 |
| pad_interior_crop_f32_n1024 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_interior_f64_n1024 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_crop_f64_n1024 | pytorch-cuda | 128 | 15 | 3.815 | 0 |
| pad_interior_crop_f64_n1024 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_interior_f32_n1048576 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_crop_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.694 | 0 |
| pad_interior_crop_f32_n1048576 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_interior_f64_n1048576 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_crop_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.555 | 0 |
| pad_interior_crop_f64_n1048576 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| reshape_copy_f32_n1024 | pytorch-cuda | 128 | 15 | 2.278 | 0 |
| stack_axis0_f32_n1024 | pytorch-cuda | 256 | 15 | 3.690 | 0 |
| stack_axis2_f32_n1024 | pytorch-cuda | 256 | 15 | 3.757 | 0 |
| reshape_copy_f64_n1024 | pytorch-cuda | 128 | 15 | 2.473 | 0 |
| stack_axis0_f64_n1024 | pytorch-cuda | 128 | 15 | 1.955 | 0 |
| stack_axis2_f64_n1024 | pytorch-cuda | 128 | 15 | 1.939 | 0 |
| reshape_copy_c32_n1024 | pytorch-cuda | 128 | 15 | 2.282 | 0 |
| stack_axis0_c32_n1024 | pytorch-cuda | 256 | 15 | 3.605 | 0 |
| stack_axis2_c32_n1024 | pytorch-cuda | 256 | 15 | 3.643 | 0 |
| reshape_copy_c64_n1024 | pytorch-cuda | 128 | 15 | 2.388 | 0 |
| stack_axis0_c64_n1024 | pytorch-cuda | 256 | 15 | 3.582 | 0 |
| stack_axis2_c64_n1024 | pytorch-cuda | 256 | 15 | 3.659 | 0 |
| reshape_copy_i32_n1024 | pytorch-cuda | 128 | 15 | 2.280 | 0 |
| stack_axis0_i32_n1024 | pytorch-cuda | 256 | 15 | 3.722 | 0 |
| stack_axis2_i32_n1024 | pytorch-cuda | 256 | 15 | 3.754 | 0 |
| reshape_copy_i64_n1024 | pytorch-cuda | 128 | 15 | 2.316 | 0 |
| stack_axis0_i64_n1024 | pytorch-cuda | 256 | 15 | 3.716 | 0 |
| stack_axis2_i64_n1024 | pytorch-cuda | 256 | 15 | 3.682 | 0 |
| reshape_copy_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.760 | 0 |
| stack_axis0_f32_n1048576 | pytorch-cuda | 32 | 15 | 1.722 | 0 |
| stack_axis2_f32_n1048576 | pytorch-cuda | 32 | 15 | 2.835 | 0 |
| reshape_copy_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.703 | 0 |
| stack_axis0_f64_n1048576 | pytorch-cuda | 16 | 15 | 1.682 | 0 |
| stack_axis2_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.780 | 0 |
| reshape_copy_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.704 | 0 |
| stack_axis0_c32_n1048576 | pytorch-cuda | 16 | 15 | 1.685 | 0 |
| stack_axis2_c32_n1048576 | pytorch-cuda | 16 | 15 | 2.778 | 0 |
| reshape_copy_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.670 | 0 |
| stack_axis0_c64_n1048576 | pytorch-cuda | 8 | 15 | 1.661 | 0 |
| stack_axis2_c64_n1048576 | pytorch-cuda | 8 | 15 | 2.781 | 0 |
| reshape_copy_i32_n1048576 | pytorch-cuda | 64 | 15 | 1.767 | 0 |
| stack_axis0_i32_n1048576 | pytorch-cuda | 32 | 15 | 1.724 | 0 |
| stack_axis2_i32_n1048576 | pytorch-cuda | 32 | 15 | 2.868 | 0 |
| reshape_copy_i64_n1048576 | pytorch-cuda | 32 | 15 | 1.699 | 0 |
| stack_axis0_i64_n1048576 | pytorch-cuda | 16 | 15 | 1.681 | 0 |
| stack_axis2_i64_n1048576 | pytorch-cuda | 16 | 15 | 2.777 | 0 |
| pad_default_f32_n1024 | jax-cuda | 32 | 15 | 3.647 | 0 |
| pad_default_f64_n1024 | jax-cuda | 16 | 15 | 2.184 | 0 |
| slice_default_f32_n1024 | jax-cuda | 16 | 15 | 2.235 | 0 |
| slice_default_f64_n1024 | jax-cuda | 16 | 15 | 2.243 | 0 |
| reverse_default_f32_n1024 | jax-cuda | 32 | 15 | 3.105 | 0 |
| reverse_default_f64_n1024 | jax-cuda | 32 | 15 | 3.021 | 0 |
| concatenate_default_f32_n1024 | jax-cuda | 16 | 15 | 2.561 | 0 |
| concatenate_default_f64_n1024 | jax-cuda | 16 | 15 | 2.490 | 0 |
| extract_diag_default_f32_n1024 | jax-cuda | 32 | 15 | 2.644 | 0 |
| extract_diag_default_f64_n1024 | jax-cuda | 32 | 15 | 2.623 | 0 |
| embed_diag_default_f32_n32 | jax-cuda | 32 | 15 | 2.753 | 0 |
| embed_diag_default_f64_n32 | jax-cuda | 32 | 15 | 2.845 | 0 |
| tril_default_f32_n1024 | jax-cuda | 32 | 15 | 2.574 | 0 |
| tril_default_f64_n1024 | jax-cuda | 32 | 15 | 2.529 | 0 |
| triu_default_f32_n1024 | jax-cuda | 32 | 15 | 2.537 | 0 |
| triu_default_f64_n1024 | jax-cuda | 32 | 15 | 2.558 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | jax-cuda | 32 | 15 | 2.634 | 0 |
| take_along_axis_batched_indices_f32_n1024 | jax-cuda | 32 | 15 | 2.657 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | jax-cuda | 32 | 15 | 2.650 | 0 |
| take_along_axis_batched_indices_f64_n1024 | jax-cuda | 32 | 15 | 2.763 | 0 |
| scatter_default_f32_n1024 | jax-cuda | 16 | 15 | 2.892 | 0 |
| scatter_default_f64_n1024 | jax-cuda | 16 | 15 | 2.698 | 0 |
| cast_default_f32_n1024 | jax-cuda | 16 | 15 | 2.265 | 0 |
| cast_default_f64_n1024 | jax-cuda | 16 | 15 | 2.275 | 0 |
| pad_default_f32_n1048576 | jax-cuda | 16 | 15 | 1.856 | 0 |
| pad_default_f64_n1048576 | jax-cuda | 16 | 15 | 1.849 | 0 |
| slice_default_f32_n1048576 | jax-cuda | 32 | 15 | 3.497 | 0 |
| slice_default_f64_n1048576 | jax-cuda | 32 | 15 | 3.405 | 0 |
| reverse_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.540 | 0 |
| reverse_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.531 | 0 |
| concatenate_default_f32_n1048576 | jax-cuda | 16 | 15 | 2.288 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda | 16 | 15 | 2.370 | 0 |
| extract_diag_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.408 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.685 | 0 |
| embed_diag_default_f32_n1024 | jax-cuda | 32 | 15 | 2.571 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda | 32 | 15 | 2.503 | 0 |
| tril_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.272 | 0 |
| tril_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.208 | 0 |
| triu_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.232 | 0 |
| triu_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.088 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | jax-cuda | 32 | 15 | 2.602 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | jax-cuda | 32 | 15 | 2.604 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | 32 | 15 | 2.671 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | jax-cuda | 32 | 15 | 2.641 | 0 |
| scatter_default_f32_n1048576 | jax-cuda | 16 | 15 | 2.550 | 0 |
| scatter_default_f64_n1048576 | jax-cuda | 16 | 15 | 2.628 | 0 |
| cast_default_f32_n1048576 | jax-cuda | 16 | 15 | 2.060 | 0 |
| cast_default_f64_n1048576 | jax-cuda | 16 | 15 | 1.998 | 0 |
| broadcast_in_dim_f32_n1024 | jax-cuda | 32 | 15 | 3.244 | 0 |
| broadcast_in_dim_f64_n1024 | jax-cuda | 32 | 15 | 3.254 | 0 |
| convert_f32_f64_n1024 | jax-cuda | 32 | 15 | 3.820 | 0 |
| broadcast_in_dim_f32_n1048576 | jax-cuda | 32 | 15 | 3.385 | 0 |
| broadcast_in_dim_f64_n1048576 | jax-cuda | 32 | 15 | 3.351 | 0 |
| convert_f32_f64_n1048576 | jax-cuda | 16 | 15 | 2.013 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | jax-cuda | 16 | 15 | 2.407 | 0 |
| dynamic_slice_clamped_f32_n1024 | jax-cuda | 16 | 15 | 2.038 | 0 |
| gather_axis0_f32_n1024 | jax-cuda | 16 | 15 | 1.976 | 0 |
| gather_axis1_f32_n1024 | jax-cuda | 16 | 15 | 2.158 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | jax-cuda | 16 | 15 | 2.486 | 0 |
| dynamic_slice_clamped_f64_n1024 | jax-cuda | 16 | 15 | 2.071 | 0 |
| gather_axis0_f64_n1024 | jax-cuda | 16 | 15 | 2.044 | 0 |
| gather_axis1_f64_n1024 | jax-cuda | 16 | 15 | 2.196 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | jax-cuda | 16 | 15 | 2.479 | 0 |
| dynamic_slice_clamped_f32_n1048576 | jax-cuda | 16 | 15 | 2.017 | 0 |
| gather_axis0_f32_n1048576 | jax-cuda | 32 | 15 | 3.697 | 0 |
| gather_axis1_f32_n1048576 | jax-cuda | 16 | 15 | 1.915 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda | 16 | 15 | 2.201 | 0 |
| dynamic_slice_clamped_f64_n1048576 | jax-cuda | 16 | 15 | 2.069 | 0 |
| gather_axis0_f64_n1048576 | jax-cuda | 32 | 15 | 3.681 | 0 |
| gather_axis1_f64_n1048576 | jax-cuda | 16 | 15 | 2.062 | 0 |
| pad_interior_f32_n1024 | jax-cuda | 32 | 15 | 3.438 | 0 |
| pad_crop_f32_n1024 | jax-cuda | 32 | 15 | 3.634 | 0 |
| pad_interior_crop_f32_n1024 | jax-cuda | 32 | 15 | 3.727 | 0 |
| pad_interior_f64_n1024 | jax-cuda | 16 | 15 | 2.173 | 0 |
| pad_crop_f64_n1024 | jax-cuda | 16 | 15 | 2.390 | 0 |
| pad_interior_crop_f64_n1024 | jax-cuda | 16 | 15 | 2.732 | 0 |
| pad_interior_f32_n1048576 | jax-cuda | 10 | 15 | 1.164 | 0 |
| pad_crop_f32_n1048576 | jax-cuda | 32 | 15 | 3.457 | 0 |
| pad_interior_crop_f32_n1048576 | jax-cuda | 10 | 15 | 1.131 | 0 |
| pad_interior_f64_n1048576 | jax-cuda | 5 | 15 | 1.097 | 0 |
| pad_crop_f64_n1048576 | jax-cuda | 32 | 15 | 3.452 | 0 |
| pad_interior_crop_f64_n1048576 | jax-cuda | 5 | 15 | 1.075 | 0 |
| reshape_copy_f32_n1024 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_f32_n1024 | jax-cuda | 16 | 15 | 2.227 | 0 |
| stack_axis2_f32_n1024 | jax-cuda | 16 | 15 | 2.333 | 0 |
| reshape_copy_f64_n1024 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_f64_n1024 | jax-cuda | 16 | 15 | 2.491 | 0 |
| stack_axis2_f64_n1024 | jax-cuda | 16 | 15 | 2.498 | 0 |
| reshape_copy_c32_n1024 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_c32_n1024 | jax-cuda | 16 | 15 | 2.582 | 0 |
| stack_axis2_c32_n1024 | jax-cuda | 16 | 15 | 2.525 | 0 |
| reshape_copy_c64_n1024 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_c64_n1024 | jax-cuda | 16 | 15 | 2.580 | 0 |
| stack_axis2_c64_n1024 | jax-cuda | 16 | 15 | 2.553 | 0 |
| reshape_copy_i32_n1024 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_i32_n1024 | jax-cuda | 16 | 15 | 2.628 | 0 |
| stack_axis2_i32_n1024 | jax-cuda | 16 | 15 | 2.358 | 0 |
| reshape_copy_i64_n1024 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_i64_n1024 | jax-cuda | 16 | 15 | 2.194 | 0 |
| stack_axis2_i64_n1024 | jax-cuda | 16 | 15 | 2.177 | 0 |
| reshape_copy_f32_n1048576 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_f32_n1048576 | jax-cuda | 16 | 15 | 2.241 | 0 |
| stack_axis2_f32_n1048576 | jax-cuda | 16 | 15 | 2.273 | 0 |
| reshape_copy_f64_n1048576 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_f64_n1048576 | jax-cuda | 16 | 15 | 2.301 | 0 |
| stack_axis2_f64_n1048576 | jax-cuda | 16 | 15 | 2.261 | 0 |
| reshape_copy_c32_n1048576 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_c32_n1048576 | jax-cuda | 16 | 15 | 2.483 | 0 |
| stack_axis2_c32_n1048576 | jax-cuda | 16 | 15 | 2.373 | 0 |
| reshape_copy_c64_n1048576 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_c64_n1048576 | jax-cuda | 8 | 15 | 2.016 | 0 |
| stack_axis2_c64_n1048576 | jax-cuda | 8 | 15 | 1.945 | 0 |
| reshape_copy_i32_n1048576 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_i32_n1048576 | jax-cuda | 16 | 15 | 2.361 | 0 |
| stack_axis2_i32_n1048576 | jax-cuda | 16 | 15 | 2.480 | 0 |
| reshape_copy_i64_n1048576 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_i64_n1048576 | jax-cuda | 16 | 15 | 2.709 | 0 |
| stack_axis2_i64_n1048576 | jax-cuda | 16 | 15 | 2.643 | 0 |
| pad_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.739 | 0 |
| pad_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.673 | 0 |
| slice_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.676 | 0 |
| slice_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.565 | 0 |
| reverse_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.555 | 0 |
| reverse_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.379 | 0 |
| concatenate_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.479 | 0 |
| concatenate_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.554 | 0 |
| extract_diag_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.394 | 0 |
| extract_diag_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.481 | 0 |
| embed_diag_default_f32_n32 | jax-cuda-jit | 32 | 15 | 2.488 | 0 |
| embed_diag_default_f64_n32 | jax-cuda-jit | 32 | 15 | 2.241 | 0 |
| tril_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.403 | 0 |
| tril_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.591 | 0 |
| triu_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.431 | 0 |
| triu_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.576 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.667 | 0 |
| take_along_axis_batched_indices_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.683 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.659 | 0 |
| take_along_axis_batched_indices_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.739 | 0 |
| scatter_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.053 | 0 |
| scatter_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.945 | 0 |
| cast_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.458 | 0 |
| cast_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.402 | 0 |
| pad_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.339 | 0 |
| pad_default_f64_n1048576 | jax-cuda-jit | 31 | 15 | 2.100 | 0 |
| slice_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.319 | 0 |
| slice_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.348 | 0 |
| reverse_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.294 | 0 |
| reverse_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.164 | 0 |
| concatenate_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.227 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.837 | 0 |
| extract_diag_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.325 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.406 | 0 |
| embed_diag_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.298 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.344 | 0 |
| tril_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.276 | 0 |
| tril_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.207 | 0 |
| triu_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.368 | 0 |
| triu_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.310 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.660 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.437 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.684 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.695 | 0 |
| scatter_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.680 | 0 |
| scatter_default_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.278 | 0 |
| cast_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.185 | 0 |
| cast_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.147 | 0 |
| broadcast_in_dim_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.840 | 0 |
| broadcast_in_dim_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.745 | 0 |
| convert_f32_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.631 | 0 |
| broadcast_in_dim_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.520 | 0 |
| broadcast_in_dim_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.571 | 0 |
| convert_f32_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.506 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.872 | 0 |
| dynamic_slice_clamped_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.872 | 0 |
| gather_axis0_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.819 | 0 |
| gather_axis1_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.797 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.803 | 0 |
| dynamic_slice_clamped_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.857 | 0 |
| gather_axis0_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.758 | 0 |
| gather_axis1_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.787 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.864 | 0 |
| dynamic_slice_clamped_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.694 | 0 |
| gather_axis0_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.673 | 0 |
| gather_axis1_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.678 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.706 | 0 |
| dynamic_slice_clamped_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.686 | 0 |
| gather_axis0_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.665 | 0 |
| gather_axis1_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.628 | 0 |
| pad_interior_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.700 | 0 |
| pad_crop_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.780 | 0 |
| pad_interior_crop_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.904 | 0 |
| pad_interior_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.098 | 0 |
| pad_crop_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.253 | 0 |
| pad_interior_crop_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.223 | 0 |
| pad_interior_f32_n1048576 | jax-cuda-jit | 10 | 15 | 1.035 | 0 |
| pad_crop_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.564 | 0 |
| pad_interior_crop_f32_n1048576 | jax-cuda-jit | 10 | 15 | 1.041 | 0 |
| pad_interior_f64_n1048576 | jax-cuda-jit | 5 | 15 | 1.037 | 0 |
| pad_crop_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.349 | 0 |
| pad_interior_crop_f64_n1048576 | jax-cuda-jit | 5 | 15 | 1.071 | 0 |
| reshape_copy_f32_n1024 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.720 | 0 |
| stack_axis2_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.609 | 0 |
| reshape_copy_f64_n1024 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.734 | 0 |
| stack_axis2_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.788 | 0 |
| reshape_copy_c32_n1024 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.794 | 0 |
| stack_axis2_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.817 | 0 |
| reshape_copy_c64_n1024 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.667 | 0 |
| stack_axis2_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.788 | 0 |
| reshape_copy_i32_n1024 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.822 | 0 |
| stack_axis2_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.777 | 0 |
| reshape_copy_i64_n1024 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.797 | 0 |
| stack_axis2_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.671 | 0 |
| reshape_copy_f32_n1048576 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.511 | 0 |
| stack_axis2_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.433 | 0 |
| reshape_copy_f64_n1048576 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.766 | 0 |
| stack_axis2_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.764 | 0 |
| reshape_copy_c32_n1048576 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_c32_n1048576 | jax-cuda-jit | 16 | 15 | 1.766 | 0 |
| stack_axis2_c32_n1048576 | jax-cuda-jit | 16 | 15 | 1.766 | 0 |
| reshape_copy_c64_n1048576 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_c64_n1048576 | jax-cuda-jit | 8 | 15 | 1.763 | 0 |
| stack_axis2_c64_n1048576 | jax-cuda-jit | 8 | 15 | 1.782 | 0 |
| reshape_copy_i32_n1048576 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_i32_n1048576 | jax-cuda-jit | 32 | 15 | 2.484 | 0 |
| stack_axis2_i32_n1048576 | jax-cuda-jit | 32 | 15 | 2.565 | 0 |
| reshape_copy_i64_n1048576 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_i64_n1048576 | jax-cuda-jit | 16 | 15 | 1.778 | 0 |
| stack_axis2_i64_n1048576 | jax-cuda-jit | 16 | 15 | 1.783 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/structural
suite_file: benchmarks/gpu/structural.yaml
timestamp: '2026-10-13T21:19:55Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: e04d33dd2ab9f4e392b8cd63d6df2b43efdb54cb
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 49f248a55b9080907f84efc5a19edeef4c74de31
  dirty: false
environment:
  hostname: 737f43342de3
  os: Linux-6.8.0-146-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '1'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    OPENBLAS_ROOT: /opt/openblas
    OPENBLAS_NUM_THREADS: '1'
    CUDA_HOME: /usr/local/cuda
    CUDA_PATH: /usr/local/cuda
    TENFERRO_CUTENSOR_PATH: /usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2
    BENCH_RUNS: '15'
    BENCH_WARMUPS: '3'
    USE_CUDA: '1'
timing_policy:
  version: 2
  scope: operation execution and completion synchronization
  setup: fixtures, H2D, runtime/session/handles, plans and compilation outside every
    timer
  cleanup: returned outputs retained until after timer stops
  minimum_untimed_priming_runs: 1
  short_operation_sampling: adaptive batches target 2 ms, at most 4096 operations
    and 256 MiB retained output; output-reuse destinations prepared before timing
  trace_execution: 'tenferro trace excluded: run_prepared cannot borrow the entered
    session; JAX JIT compilation outside timing'
  setup_diagnostics: false
blas:
  implementation: none
python_backends:
  pytorch:
    available: true
    version: 2.12.0+cu130
    file: /workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/__init__.py
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
    library: /workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so
    linked_libraries:
    - /workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1
  jax:
    available: true
    version: 0.10.1
    jaxlib_version: 0.10.1
    backend: gpu
    file: /workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/jaxlib/__init__.py
    provider: xla_gpu
    dot_backend: xla_gpu
    lapack_provider: none_detected
    library: /workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/jaxlib/_ifrt_proxy.so
    linked_libraries: []
julia:
  available: false
  reason: julia not found on PATH
cuda:
  device_ordinal: 0
  device_name: NVIDIA GeForce RTX 3060
  device_uuid: GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e
  device_memory_bytes: 12884901888
  driver_version: 580.178.04
  runtime_version: '12.9'
  cuda_version: '13.0'
  cudnn_version: '92700'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e
  cuda_visible_devices: GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh structural
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 118
  selected_cases: 118
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: 19d46353cb2f682dcda76ef3aa8d7733f39d4843aa38ef6231fd496ec575d6a9
    src/bin/benchmark_gpu_public_api/plan.rs: 3d0f10f390ebd0e08a40deb5a64e49e086dcd41b9a777d3a208c6846ed6b2fac
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: da168bf8b7c30f748f62839cf0bf7cce8f5bccf9c37b1c1d993fd70c466f2f76
    src/bin/benchmark_gpu_public_api/multi.rs: 1b477fef78260cf28aa31e515d57ab411f6241b91ffb30d625939538fdbc6bab
    src/bin/benchmark_gpu_public_api/lifetime.rs: c687a8b0329f6d3073073671b5d246acbb1f4dc3f983d1430da897a908ef1745
    src/bin/benchmark_gpu_public_api/read_layout.rs: 70b1ed632ad52db5654c3d9f3df9fdb20a1e8c48894dbdcb24ce055757f5f7eb
    src/bin/benchmark_gpu_public_api/structural.rs: 6b6c5f3c2e771d527dbab1ae4e9dd0354bd1bb88acbe4fb1292ce8b68b408240
    src/bin/benchmark_gpu_public_api/linalg.rs: 5f5c4afc161f3d6e267861511d0d5871be9192d9968c95b59a73c9dc5a64e22b
    src/bin/benchmark_gpu_public_api/state.rs: af8e9c9b4eb272af9a751d2a0aef9e1f8ff32c5efbd3ffffcf5f76e1e2c9704a
    src/bin/benchmark_gpu_public_api/reuse.rs: 1dc14ac94fac700aa696d0ada04315685e3245738f9514de3d7989b8cb59152b
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: c5f9507e8971ae14acd540d6c35b305c35b222431ec5aad9830a88bdc32975b0
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: dd5e4cc47e85fcb5afdc56d460136b2a399a6a7d6a62005de8ba94afb0119ecc
    data/results/nvidia-gpu/gpu/structural/20261013_211954/fixtures/manifest.json: 6b4532ecb67f54e60c796185be6597f7f29b0ca79754f79e16488a51241e9220
    target/release/benchmark_gpu_public_api: cffb781f4b953aacb5db9e5d0a86a57c6b088cc5161a0e5e384b322e7066a0f5
  reference_numpy_version: 2.4.2
```
