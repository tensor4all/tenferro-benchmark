# gpu/structural: CUDA public API benchmark

Measured cells: **534**. Unsupported cells: **56**, including **8** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/structural/20261102_210608/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

I32/I64 outputs require exact integer equality without conversion to floating point. Division and remainder truncate toward zero; axis-1 sum preserves the input dtype. Arithmetic fixtures avoid overflow, and I64 unary/order fixtures include values above 2**53 to detect loss of integer precision. A zero verification value in these rows means exact equality.

Reshape-copy cases add a leading singleton dimension and materialize the full output. Direct Tensor.reshape and EagerSession.reshape copy device data; PyTorch calls public view_copy. Direct/PyTorch initial/final allocation-pointer checks are untimed and reject input aliasing. Eager reshape copies its input before returning a descriptor view; original CUDA placement/shape/dtype are checked and a public inspection copy is made outside timing. JAX reshape has no established matching owned-output guarantee and remains unsupported; reshape-plus-copy composition is not substituted.

Stack selects actual public Tensor.stack/EagerSession.stack with two distinct inputs and axis 0 or 2. Their intrinsic intermediate allocations are part of this public stack operation. PyTorch/JAX use native stack; final outputs are fully verified. Direct Rust initial/final pointer checks reject aliasing either input outside timing; eager original descriptors are checked before any untimed inspection copy.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| pad_default_f32_n1024 | 19.643 (0.117) | 18.824 (0.070) | 13.275 (0.042) | 70.282 (0.430) | 40.472 (0.270) |
| pad_default_f64_n1024 | 19.227 (0.197) | 19.943 (0.647) | 13.292 (0.086) | 69.908 (0.995) | 40.583 (0.320) |
| slice_default_f32_n1024 | 16.944 (0.679) | unsupported | 9.478 (0.045) | 69.870 (0.367) | 40.574 (0.146) |
| slice_default_f64_n1024 | 18.154 (0.192) | unsupported | 9.630 (0.120) | 70.061 (0.480) | 40.403 (0.355) |
| reverse_default_f32_n1024 | 16.971 (0.138) | 16.691 (0.135) | 6.569 (0.011) | 48.677 (0.317) | 40.422 (0.224) |
| reverse_default_f64_n1024 | 17.198 (0.666) | 18.277 (0.301) | 6.462 (0.008) | 48.667 (0.208) | 40.195 (0.253) |
| concatenate_default_f32_n1024 | 22.399 (0.075) | 22.612 (0.294) | 6.296 (0.124) | 97.739 (0.320) | 42.541 (0.252) |
| concatenate_default_f64_n1024 | 22.071 (0.755) | 22.876 (1.625) | 6.411 (0.032) | 97.650 (0.584) | 42.577 (0.364) |
| extract_diag_default_f32_n1024 | 16.830 (0.456) | 18.617 (0.458) | 7.506 (0.025) | 48.096 (0.317) | 40.379 (0.274) |
| extract_diag_default_f64_n1024 | 16.717 (0.899) | 18.101 (0.049) | 7.455 (0.111) | 47.410 (0.283) | 39.755 (0.255) |
| embed_diag_default_f32_n32 | 17.415 (0.118) | 18.030 (0.182) | 11.320 (0.034) | 47.832 (0.245) | 40.145 (0.392) |
| embed_diag_default_f64_n32 | 17.940 (0.124) | 18.815 (0.289) | 11.607 (0.208) | 47.410 (0.224) | 39.697 (0.573) |
| tril_default_f32_n1024 | 15.980 (0.183) | 16.879 (0.130) | 4.991 (0.014) | 39.846 (0.363) | 39.510 (0.273) |
| tril_default_f64_n1024 | 16.777 (0.092) | 17.939 (0.085) | 5.044 (0.014) | 40.047 (0.325) | 40.058 (0.210) |
| triu_default_f32_n1024 | 16.402 (0.127) | 16.976 (0.163) | 4.963 (0.013) | 39.855 (0.281) | 39.760 (0.355) |
| triu_default_f64_n1024 | 17.100 (0.077) | 17.615 (0.073) | 5.001 (0.017) | 39.857 (0.191) | 39.602 (0.262) |
| take_along_axis_broadcast_indices_f32_n1024 | 39.625 (0.150) | 49.887 (0.167) | 18.002 (0.230) | 43.327 (0.235) | 42.654 (0.370) |
| take_along_axis_batched_indices_f32_n1024 | unsupported | unsupported | 17.801 (0.122) | 42.969 (0.318) | 42.452 (0.146) |
| take_along_axis_broadcast_indices_f64_n1024 | 39.665 (2.788) | 50.387 (0.492) | 18.291 (0.319) | 43.246 (0.315) | 41.972 (0.121) |
| take_along_axis_batched_indices_f64_n1024 | unsupported | unsupported | 17.739 (0.114) | 43.267 (0.306) | 42.303 (0.169) |
| scatter_default_f32_n1024 | 27.086 (0.845) | 31.118 (0.154) | 12.271 (0.131) | 106.970 (0.783) | 53.561 (0.363) |
| scatter_default_f64_n1024 | 28.506 (0.629) | 32.795 (0.111) | 12.630 (0.082) | 108.424 (0.593) | 53.734 (0.338) |
| cast_default_f32_n1024 | 15.035 (0.801) | 35.347 (2.425) | 6.943 (0.026) | 84.061 (0.235) | 40.184 (0.203) |
| cast_default_f64_n1024 | 15.444 (0.575) | 35.596 (0.329) | 6.915 (0.031) | 84.275 (0.671) | 40.361 (0.420) |
| pad_default_f32_n1048576 | 38.347 (0.799) | 38.712 (0.148) | 13.611 (0.071) | 72.537 (0.410) | 41.701 (0.375) |
| pad_default_f64_n1048576 | 42.786 (0.169) | 42.533 (0.198) | 15.478 (0.102) | 71.749 (0.509) | 41.979 (0.406) |
| slice_default_f32_n1048576 | 23.906 (0.058) | unsupported | 9.945 (0.058) | 69.902 (0.387) | 41.558 (0.569) |
| slice_default_f64_n1048576 | 25.333 (0.157) | unsupported | 10.165 (0.182) | 70.938 (0.380) | 41.454 (0.457) |
| reverse_default_f32_n1048576 | 26.221 (0.086) | 26.002 (0.047) | 7.697 (0.033) | 49.296 (0.231) | 41.449 (0.180) |
| reverse_default_f64_n1048576 | 30.870 (1.199) | 30.569 (0.155) | 10.808 (0.030) | 50.138 (0.323) | 42.219 (0.362) |
| concatenate_default_f32_n1048576 | 47.938 (1.114) | 48.623 (0.203) | 10.075 (0.112) | 98.440 (0.477) | 44.901 (0.413) |
| concatenate_default_f64_n1048576 | 57.951 (0.043) | 61.253 (0.053) | 25.745 (0.310) | 101.116 (0.843) | 44.738 (1.125) |
| extract_diag_default_f32_n1048576 | 16.810 (0.107) | 17.370 (0.108) | 7.492 (0.025) | 48.049 (0.271) | 40.265 (0.323) |
| extract_diag_default_f64_n1048576 | 17.581 (0.798) | 18.004 (0.622) | 7.509 (0.023) | 47.733 (0.170) | 40.096 (0.359) |
| embed_diag_default_f32_n1024 | 19.732 (0.132) | 21.594 (1.248) | 11.676 (0.076) | 49.225 (0.130) | 41.549 (0.272) |
| embed_diag_default_f64_n1024 | 28.023 (0.058) | 28.002 (0.022) | 11.888 (0.270) | 49.622 (0.511) | 41.555 (0.340) |
| tril_default_f32_n1048576 | 24.429 (0.102) | 26.475 (0.123) | 8.169 (0.042) | 41.263 (0.312) | 41.594 (0.289) |
| tril_default_f64_n1048576 | 29.167 (1.029) | 30.661 (0.177) | 10.844 (0.040) | 41.668 (0.370) | 41.412 (0.311) |
| triu_default_f32_n1048576 | 24.475 (0.816) | 26.251 (0.170) | 8.333 (0.031) | 41.355 (0.424) | 41.875 (0.170) |
| triu_default_f64_n1048576 | 29.042 (1.153) | 30.309 (0.168) | 11.043 (0.040) | 42.902 (1.436) | 42.276 (0.501) |
| take_along_axis_broadcast_indices_f32_n1048576 | 57.500 (1.100) | 68.932 (0.311) | 17.853 (0.077) | 44.642 (0.218) | 44.293 (0.146) |
| take_along_axis_batched_indices_f32_n1048576 | unsupported | unsupported | 17.504 (0.064) | 44.636 (0.319) | 43.986 (0.070) |
| take_along_axis_broadcast_indices_f64_n1048576 | 59.766 (0.294) | 68.560 (0.557) | 18.354 (0.040) | 44.830 (0.261) | 44.358 (0.185) |
| take_along_axis_batched_indices_f64_n1048576 | unsupported | unsupported | 17.752 (0.078) | 44.112 (0.211) | 44.430 (0.315) |
| scatter_default_f32_n1048576 | 56.058 (0.109) | 56.246 (0.222) | 13.808 (0.057) | 107.874 (0.473) | 54.968 (0.334) |
| scatter_default_f64_n1048576 | 63.245 (0.276) | 68.013 (0.372) | 22.759 (0.066) | 108.882 (0.629) | 55.268 (0.453) |
| cast_default_f32_n1048576 | 28.220 (0.166) | 55.205 (0.330) | 10.583 (0.045) | 86.068 (0.729) | 42.035 (0.322) |
| cast_default_f64_n1048576 | 23.856 (1.007) | 51.816 (0.330) | 9.141 (0.027) | 85.210 (0.343) | 41.605 (0.487) |
| broadcast_in_dim_f32_n1024 | 16.777 (0.757) | unsupported | 7.888 (0.010) | 69.350 (0.625) | 39.930 (0.273) |
| broadcast_in_dim_f64_n1024 | 18.208 (0.588) | unsupported | 7.814 (0.030) | 69.655 (0.391) | 40.129 (0.344) |
| convert_f32_f64_n1024 | 15.019 (1.029) | 36.389 (0.651) | 6.977 (0.016) | 84.136 (0.520) | 39.986 (0.334) |
| broadcast_in_dim_f32_n1048576 | 32.756 (0.317) | unsupported | 8.043 (0.190) | 70.262 (0.394) | 41.457 (0.201) |
| broadcast_in_dim_f64_n1048576 | 35.476 (0.760) | unsupported | 8.326 (0.180) | 71.279 (0.475) | 42.047 (0.343) |
| convert_f32_f64_n1048576 | 28.487 (0.089) | 55.294 (1.861) | 10.652 (0.067) | 85.514 (0.465) | 41.974 (0.533) |
| dynamic_slice_in_bounds_f32_n1024 | 17.437 (0.083) | 18.282 (1.024) | unsupported | 87.926 (0.466) | 44.028 (0.375) |
| dynamic_slice_clamped_f32_n1024 | 17.502 (0.038) | 18.055 (0.486) | unsupported | 88.088 (0.249) | 43.750 (0.187) |
| gather_axis0_f32_n1024 | 20.707 (0.727) | 21.343 (0.107) | 9.878 (0.011) | 78.024 (0.713) | 42.684 (0.198) |
| gather_axis1_f32_n1024 | 21.142 (0.571) | 21.337 (0.158) | 9.157 (0.008) | 78.810 (0.729) | 42.925 (0.174) |
| dynamic_slice_in_bounds_f64_n1024 | 16.406 (0.073) | 18.191 (0.089) | unsupported | 88.588 (0.340) | 43.784 (0.411) |
| dynamic_slice_clamped_f64_n1024 | 16.484 (0.804) | 18.057 (0.131) | unsupported | 88.703 (0.385) | 43.725 (0.348) |
| gather_axis0_f64_n1024 | 21.069 (0.197) | 21.525 (0.099) | 9.818 (0.010) | 77.856 (0.376) | 43.087 (0.412) |
| gather_axis1_f64_n1024 | 21.352 (0.139) | 21.336 (0.073) | 9.784 (0.008) | 78.555 (0.628) | 43.127 (0.320) |
| dynamic_slice_in_bounds_f32_n1048576 | 21.486 (0.098) | 20.872 (0.038) | unsupported | 88.356 (0.237) | 44.776 (0.217) |
| dynamic_slice_clamped_f32_n1048576 | 21.283 (0.112) | 20.900 (0.134) | unsupported | 88.804 (0.326) | 44.510 (0.164) |
| gather_axis0_f32_n1048576 | 38.853 (0.189) | 41.969 (0.168) | 9.810 (0.158) | 78.196 (0.644) | 44.101 (0.284) |
| gather_axis1_f32_n1048576 | 38.905 (0.088) | 42.027 (0.142) | 10.051 (0.087) | 79.147 (0.397) | 44.379 (0.160) |
| dynamic_slice_in_bounds_f64_n1048576 | 21.257 (0.839) | 24.315 (0.517) | unsupported | 88.621 (0.604) | 44.905 (0.230) |
| dynamic_slice_clamped_f64_n1048576 | 21.198 (0.449) | 24.250 (0.541) | unsupported | 88.690 (0.203) | 45.377 (0.405) |
| gather_axis0_f64_n1048576 | 42.382 (0.598) | 44.516 (0.131) | 9.874 (0.103) | 80.145 (0.367) | 44.634 (0.149) |
| gather_axis1_f64_n1048576 | 41.211 (0.160) | 44.904 (0.191) | 10.055 (0.159) | 79.918 (0.266) | 44.157 (0.168) |
| pad_interior_f32_n1024 | 18.461 (1.043) | 20.592 (0.122) | unsupported | 71.245 (0.206) | 40.082 (0.549) |
| pad_crop_f32_n1024 | 18.848 (1.291) | 19.857 (0.200) | 12.486 (0.745) | 71.059 (0.265) | 40.005 (0.345) |
| pad_interior_crop_f32_n1024 | 18.312 (0.093) | 20.240 (0.400) | unsupported | 71.638 (0.386) | 39.966 (0.339) |
| pad_interior_f64_n1024 | 19.532 (0.099) | 20.548 (0.104) | unsupported | 71.700 (0.260) | 40.046 (0.296) |
| pad_crop_f64_n1024 | 19.714 (0.093) | 20.600 (0.062) | 12.553 (0.125) | 71.408 (0.430) | 39.711 (0.493) |
| pad_interior_crop_f64_n1024 | 19.694 (0.091) | 20.629 (0.370) | unsupported | 71.825 (0.754) | 39.670 (0.273) |
| pad_interior_f32_n1048576 | 126.923 (0.376) | 127.705 (0.370) | unsupported | 77.946 (0.749) | 44.451 (1.272) |
| pad_crop_f32_n1048576 | 38.804 (0.299) | 38.360 (0.807) | 12.569 (0.089) | 72.631 (0.548) | 35.286 (2.143) |
| pad_interior_crop_f32_n1048576 | 126.359 (0.097) | 127.053 (0.523) | unsupported | 77.942 (0.903) | 36.293 (1.746) |
| pad_interior_f64_n1048576 | 142.929 (0.248) | 203.017 (0.117) | unsupported | 87.241 (2.802) | 60.814 (2.734) |
| pad_crop_f64_n1048576 | 42.385 (0.244) | 42.178 (0.202) | 15.821 (0.081) | 72.619 (0.603) | 35.615 (7.093) |
| pad_interior_crop_f64_n1048576 | 142.589 (0.392) | 143.623 (0.544) | unsupported | 86.979 (1.827) | 70.096 (0.526) |
| reshape_copy_f32_n1024 | 15.881 (0.118) | 17.711 (0.148) | 7.694 (1.071) | unsupported | unsupported |
| stack_axis0_f32_n1024 | 57.918 (0.797) | 63.111 (0.272) | 6.159 (0.328) | 81.562 (1.736) | 36.103 (4.794) |
| stack_axis2_f32_n1024 | 54.640 (2.340) | 68.291 (1.047) | 7.075 (0.616) | 81.976 (7.637) | 42.177 (0.747) |
| reshape_copy_f64_n1024 | 16.110 (0.376) | 19.692 (0.190) | 6.819 (0.241) | unsupported | unsupported |
| stack_axis0_f64_n1024 | 55.172 (1.541) | 64.805 (3.367) | 6.260 (0.398) | 82.070 (2.969) | 42.018 (0.878) |
| stack_axis2_f64_n1024 | 57.029 (0.276) | 70.326 (0.388) | 6.458 (0.885) | 84.167 (9.915) | 37.672 (5.523) |
| reshape_copy_c32_n1024 | 15.380 (0.487) | 19.711 (0.051) | 7.212 (1.233) | unsupported | unsupported |
| stack_axis0_c32_n1024 | 56.355 (2.087) | 69.368 (0.408) | 7.317 (0.795) | 97.931 (5.733) | 36.617 (3.680) |
| stack_axis2_c32_n1024 | 57.210 (0.169) | 69.446 (0.966) | 7.779 (0.920) | 97.969 (2.410) | 36.204 (0.121) |
| reshape_copy_c64_n1024 | 16.040 (1.017) | 20.212 (0.368) | 8.692 (0.599) | unsupported | unsupported |
| stack_axis0_c64_n1024 | 54.553 (0.302) | 70.041 (0.845) | 7.395 (0.550) | 89.203 (11.073) | 42.500 (6.406) |
| stack_axis2_c64_n1024 | 54.238 (0.236) | 70.806 (0.420) | 7.303 (0.434) | 98.122 (16.273) | 42.872 (0.344) |
| reshape_copy_i32_n1024 | 14.571 (0.431) | 19.460 (0.125) | 7.971 (0.182) | unsupported | unsupported |
| stack_axis0_i32_n1024 | 54.332 (2.587) | 68.222 (0.483) | 7.236 (0.433) | 84.092 (8.647) | 39.808 (5.413) |
| stack_axis2_i32_n1024 | 54.591 (3.705) | 68.478 (0.433) | 7.526 (0.463) | 88.127 (8.253) | 42.815 (0.279) |
| reshape_copy_i64_n1024 | 16.197 (1.185) | 20.260 (0.430) | 8.493 (1.388) | unsupported | unsupported |
| stack_axis0_i64_n1024 | 54.828 (1.866) | 70.313 (0.524) | 7.634 (0.638) | 88.457 (14.722) | 42.543 (0.360) |
| stack_axis2_i64_n1024 | 57.793 (0.169) | 69.345 (0.411) | 6.634 (0.634) | 96.676 (10.264) | 42.640 (0.658) |
| reshape_copy_f32_n1048576 | 23.072 (0.189) | 29.050 (0.131) | 6.942 (0.115) | unsupported | unsupported |
| stack_axis0_f32_n1048576 | 97.996 (0.283) | 114.655 (0.397) | 8.874 (0.114) | 82.685 (0.498) | 44.536 (0.386) |
| stack_axis2_f32_n1048576 | 98.823 (0.285) | 114.579 (0.460) | 25.876 (0.150) | 98.141 (5.512) | 44.517 (0.233) |
| reshape_copy_f64_n1048576 | 28.202 (0.071) | 33.370 (0.095) | 9.251 (0.921) | unsupported | unsupported |
| stack_axis0_f64_n1048576 | 123.566 (0.530) | 136.989 (0.948) | 25.293 (0.252) | 85.846 (0.557) | 41.565 (6.772) |
| stack_axis2_f64_n1048576 | 122.952 (0.992) | 136.916 (1.734) | 45.865 (0.220) | 100.793 (0.373) | 43.500 (5.813) |
| reshape_copy_c32_n1048576 | 28.338 (0.131) | 31.769 (0.088) | 8.734 (0.091) | unsupported | unsupported |
| stack_axis0_c32_n1048576 | 123.862 (0.468) | 142.585 (0.447) | 25.379 (0.245) | 101.158 (0.830) | 37.779 (1.325) |
| stack_axis2_c32_n1048576 | 123.057 (0.383) | 140.625 (4.816) | 46.058 (0.158) | 101.222 (1.002) | 45.202 (1.061) |
| reshape_copy_c64_n1048576 | 57.099 (0.050) | 60.164 (0.055) | 24.948 (0.301) | unsupported | unsupported |
| stack_axis0_c64_n1048576 | 385.545 (1.476) | 412.888 (0.879) | 47.388 (0.339) | 98.927 (16.402) | 57.463 (0.532) |
| stack_axis2_c64_n1048576 | 321.483 (4.294) | 392.938 (2.507) | 101.290 (0.311) | 109.366 (0.991) | 65.252 (3.610) |
| reshape_copy_i32_n1048576 | 21.965 (0.073) | 27.272 (0.112) | 6.919 (0.148) | unsupported | unsupported |
| stack_axis0_i32_n1048576 | 98.479 (1.858) | 116.852 (2.820) | 8.862 (0.074) | 98.061 (0.372) | 49.327 (3.601) |
| stack_axis2_i32_n1048576 | 101.138 (1.278) | 113.349 (0.473) | 25.655 (0.129) | 99.054 (0.757) | 48.783 (5.017) |
| reshape_copy_i64_n1048576 | 28.168 (0.021) | 33.640 (0.331) | 8.694 (0.150) | unsupported | unsupported |
| stack_axis0_i64_n1048576 | 120.074 (0.436) | 144.228 (0.347) | 25.238 (0.247) | 100.321 (0.500) | 47.012 (4.934) |
| stack_axis2_i64_n1048576 | 120.045 (0.341) | 143.352 (0.994) | 45.981 (0.209) | 100.263 (0.921) | 46.641 (2.788) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| pad_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.514 | 0 |
| pad_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.461 | 0 |
| slice_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.169 | 0 |
| slice_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.324 | 0 |
| reverse_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.172 | 0 |
| reverse_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.201 | 0 |
| concatenate_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.867 | 0 |
| concatenate_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.825 | 0 |
| extract_diag_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.154 | 0 |
| extract_diag_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.140 | 0 |
| embed_diag_default_f32_n32 | tenferro-cuda-direct | 128 | 15 | 2.229 | 0 |
| embed_diag_default_f64_n32 | tenferro-cuda-direct | 128 | 15 | 2.296 | 0 |
| tril_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.045 | 0 |
| tril_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.147 | 0 |
| triu_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.099 | 0 |
| triu_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.189 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.536 | 0 |
| take_along_axis_batched_indices_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.539 | 0 |
| take_along_axis_batched_indices_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.467 | 0 |
| scatter_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.649 | 0 |
| cast_default_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.849 | 0 |
| cast_default_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.954 | 0 |
| pad_default_f32_n1048576 | tenferro-cuda-direct | 63 | 15 | 2.416 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-direct | 31 | 15 | 1.326 | 0 |
| slice_default_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 3.060 | 0 |
| slice_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.621 | 0 |
| reverse_default_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.678 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.988 | 0 |
| concatenate_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.534 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.927 | 0 |
| extract_diag_default_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.152 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.250 | 0 |
| embed_diag_default_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.263 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 0.897 | 0 |
| tril_default_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.563 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.933 | 0 |
| triu_default_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.566 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.929 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.680 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.913 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.588 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.024 | 0 |
| cast_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.903 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.527 | 0 |
| broadcast_in_dim_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.148 | 0 |
| broadcast_in_dim_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.331 | 0 |
| convert_f32_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 1.922 | 0 |
| broadcast_in_dim_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.096 | 0 |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.135 | 0 |
| convert_f32_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.912 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.232 | 0 |
| dynamic_slice_clamped_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.240 | 0 |
| gather_axis0_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.651 | 0 |
| gather_axis1_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.706 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.100 | 0 |
| dynamic_slice_clamped_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.110 | 0 |
| gather_axis0_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.697 | 0 |
| gather_axis1_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.733 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.750 | 0 |
| dynamic_slice_clamped_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.724 | 0 |
| gather_axis0_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.487 | 0 |
| gather_axis1_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.490 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.721 | 0 |
| dynamic_slice_clamped_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.713 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.712 | 0 |
| gather_axis1_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.638 | 0 |
| pad_interior_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.363 | 0 |
| pad_crop_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.412 | 0 |
| pad_interior_crop_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.344 | 0 |
| pad_interior_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.500 | 0 |
| pad_crop_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.523 | 0 |
| pad_interior_crop_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.521 | 0 |
| pad_interior_f32_n1048576 | tenferro-cuda-direct | 10 | 15 | 1.269 | 0 |
| pad_crop_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.483 | 0 |
| pad_interior_crop_f32_n1048576 | tenferro-cuda-direct | 10 | 15 | 1.264 | 0 |
| pad_interior_f64_n1048576 | tenferro-cuda-direct | 5 | 15 | 0.715 | 0 |
| pad_crop_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.356 | 0 |
| pad_interior_crop_f64_n1048576 | tenferro-cuda-direct | 5 | 15 | 0.713 | 0 |
| reshape_copy_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.033 | 0 |
| stack_axis0_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.707 | 0 |
| stack_axis2_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.497 | 0 |
| reshape_copy_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.062 | 0 |
| stack_axis0_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.531 | 0 |
| stack_axis2_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.650 | 0 |
| reshape_copy_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.937 | 0 |
| stack_axis0_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.607 | 0 |
| stack_axis2_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.661 | 0 |
| reshape_copy_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.053 | 0 |
| stack_axis0_c64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.491 | 0 |
| stack_axis2_c64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.471 | 0 |
| reshape_copy_i32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.730 | 0 |
| stack_axis0_i32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.477 | 0 |
| stack_axis2_i32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.494 | 0 |
| reshape_copy_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.073 | 0 |
| stack_axis0_i64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.509 | 0 |
| stack_axis2_i64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.699 | 0 |
| reshape_copy_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.477 | 0 |
| stack_axis0_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.568 | 0 |
| stack_axis2_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.581 | 0 |
| reshape_copy_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.902 | 0 |
| stack_axis0_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.977 | 0 |
| stack_axis2_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.967 | 0 |
| reshape_copy_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.907 | 0 |
| stack_axis0_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.982 | 0 |
| stack_axis2_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.969 | 0 |
| reshape_copy_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.914 | 0 |
| stack_axis0_c64_n1048576 | tenferro-cuda-direct | 8 | 15 | 3.084 | 0 |
| stack_axis2_c64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.572 | 0 |
| reshape_copy_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.406 | 0 |
| stack_axis0_i32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.151 | 0 |
| stack_axis2_i32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.236 | 0 |
| reshape_copy_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.901 | 0 |
| stack_axis0_i64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.921 | 0 |
| stack_axis2_i64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.921 | 0 |
| pad_default_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.410 | 0 |
| pad_default_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.553 | 0 |
| slice_default_f32_n1024 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| slice_default_f64_n1024 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| reverse_default_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.136 | 0 |
| reverse_default_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.339 | 0 |
| concatenate_default_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.894 | 0 |
| concatenate_default_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.928 | 0 |
| extract_diag_default_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.383 | 0 |
| extract_diag_default_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.317 | 0 |
| embed_diag_default_f32_n32 | tenferro-cuda-eager | 128 | 15 | 2.308 | 0 |
| embed_diag_default_f64_n32 | tenferro-cuda-eager | 128 | 15 | 2.408 | 0 |
| tril_default_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.161 | 0 |
| tril_default_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.296 | 0 |
| triu_default_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.173 | 0 |
| triu_default_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.255 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.193 | 0 |
| take_along_axis_batched_indices_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.225 | 0 |
| take_along_axis_batched_indices_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.983 | 0 |
| scatter_default_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.099 | 0 |
| cast_default_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.262 | 0 |
| cast_default_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.278 | 0 |
| pad_default_f32_n1048576 | tenferro-cuda-eager | 63 | 15 | 2.439 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-eager | 31 | 15 | 1.319 | 0 |
| slice_default_f32_n1048576 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| slice_default_f64_n1048576 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| reverse_default_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.664 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.978 | 0 |
| concatenate_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.556 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.980 | 0 |
| extract_diag_default_f32_n1048576 | tenferro-cuda-eager | 128 | 15 | 2.223 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | 128 | 15 | 2.305 | 0 |
| embed_diag_default_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 1.382 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 0.896 | 0 |
| tril_default_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.694 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.981 | 0 |
| triu_default_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.680 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.970 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.206 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.194 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.800 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.176 | 0 |
| cast_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.767 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.658 | 0 |
| broadcast_in_dim_f32_n1024 | tenferro-cuda-eager | — | — | — | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| broadcast_in_dim_f64_n1024 | tenferro-cuda-eager | — | — | — | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| convert_f32_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.329 | 0 |
| broadcast_in_dim_f32_n1048576 | tenferro-cuda-eager | — | — | — | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-eager | — | — | — | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| convert_f32_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.769 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.340 | 0 |
| dynamic_slice_clamped_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.311 | 0 |
| gather_axis0_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.732 | 0 |
| gather_axis1_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.731 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.328 | 0 |
| dynamic_slice_clamped_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.311 | 0 |
| gather_axis0_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.755 | 0 |
| gather_axis1_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.731 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | tenferro-cuda-eager | 128 | 15 | 2.672 | 0 |
| dynamic_slice_clamped_f32_n1048576 | tenferro-cuda-eager | 128 | 15 | 2.675 | 0 |
| gather_axis0_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.686 | 0 |
| gather_axis1_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.690 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-eager | 128 | 15 | 3.112 | 0 |
| dynamic_slice_clamped_f64_n1048576 | tenferro-cuda-eager | 128 | 15 | 3.104 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.849 | 0 |
| gather_axis1_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.874 | 0 |
| pad_interior_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.636 | 0 |
| pad_crop_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.542 | 0 |
| pad_interior_crop_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.591 | 0 |
| pad_interior_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.630 | 0 |
| pad_crop_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.637 | 0 |
| pad_interior_crop_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.641 | 0 |
| pad_interior_f32_n1048576 | tenferro-cuda-eager | 10 | 15 | 1.277 | 0 |
| pad_crop_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.455 | 0 |
| pad_interior_crop_f32_n1048576 | tenferro-cuda-eager | 10 | 15 | 1.271 | 0 |
| pad_interior_f64_n1048576 | tenferro-cuda-eager | 5 | 15 | 1.015 | 0 |
| pad_crop_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.350 | 0 |
| pad_interior_crop_f64_n1048576 | tenferro-cuda-eager | 5 | 15 | 0.718 | 0 |
| reshape_copy_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.267 | 0 |
| stack_axis0_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.020 | 0 |
| stack_axis2_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.185 | 0 |
| reshape_copy_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.521 | 0 |
| stack_axis0_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.074 | 0 |
| stack_axis2_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.250 | 0 |
| reshape_copy_c32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.523 | 0 |
| stack_axis0_c32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.220 | 0 |
| stack_axis2_c32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.222 | 0 |
| reshape_copy_c64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.587 | 0 |
| stack_axis0_c64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.241 | 0 |
| stack_axis2_c64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.266 | 0 |
| reshape_copy_i32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.491 | 0 |
| stack_axis0_i32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.183 | 0 |
| stack_axis2_i32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.191 | 0 |
| reshape_copy_i64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.593 | 0 |
| stack_axis0_i64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.250 | 0 |
| stack_axis2_i64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.219 | 0 |
| reshape_copy_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.859 | 0 |
| stack_axis0_f32_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.834 | 0 |
| stack_axis2_f32_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.833 | 0 |
| reshape_copy_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.068 | 0 |
| stack_axis0_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.192 | 0 |
| stack_axis2_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.191 | 0 |
| reshape_copy_c32_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.017 | 0 |
| stack_axis0_c32_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.281 | 0 |
| stack_axis2_c32_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.250 | 0 |
| reshape_copy_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.963 | 0 |
| stack_axis0_c64_n1048576 | tenferro-cuda-eager | 8 | 15 | 3.303 | 0 |
| stack_axis2_c64_n1048576 | tenferro-cuda-eager | 8 | 15 | 3.144 | 0 |
| reshape_copy_i32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.745 | 0 |
| stack_axis0_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.739 | 0 |
| stack_axis2_i32_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.814 | 0 |
| reshape_copy_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.076 | 0 |
| stack_axis0_i64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.308 | 0 |
| stack_axis2_i64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.294 | 0 |
| pad_default_f32_n1024 | pytorch-cuda | 256 | 15 | 3.398 | 0 |
| pad_default_f64_n1024 | pytorch-cuda | 256 | 15 | 3.403 | 0 |
| slice_default_f32_n1024 | pytorch-cuda | 256 | 15 | 2.426 | 0 |
| slice_default_f64_n1024 | pytorch-cuda | 256 | 15 | 2.465 | 0 |
| reverse_default_f32_n1024 | pytorch-cuda | 512 | 15 | 3.363 | 0 |
| reverse_default_f64_n1024 | pytorch-cuda | 512 | 15 | 3.308 | 0 |
| concatenate_default_f32_n1024 | pytorch-cuda | 512 | 15 | 3.223 | 0 |
| concatenate_default_f64_n1024 | pytorch-cuda | 512 | 15 | 3.282 | 0 |
| extract_diag_default_f32_n1024 | pytorch-cuda | 512 | 15 | 3.843 | 0 |
| extract_diag_default_f64_n1024 | pytorch-cuda | 512 | 15 | 3.817 | 0 |
| embed_diag_default_f32_n32 | pytorch-cuda | 256 | 15 | 2.898 | 0 |
| embed_diag_default_f64_n32 | pytorch-cuda | 256 | 15 | 2.971 | 0 |
| tril_default_f32_n1024 | pytorch-cuda | 512 | 15 | 2.555 | 0 |
| tril_default_f64_n1024 | pytorch-cuda | 512 | 15 | 2.583 | 0 |
| triu_default_f32_n1024 | pytorch-cuda | 512 | 15 | 2.541 | 0 |
| triu_default_f64_n1024 | pytorch-cuda | 512 | 15 | 2.560 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | pytorch-cuda | 128 | 15 | 2.304 | 0 |
| take_along_axis_batched_indices_f32_n1024 | pytorch-cuda | 128 | 15 | 2.279 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | pytorch-cuda | 128 | 15 | 2.341 | 0 |
| take_along_axis_batched_indices_f64_n1024 | pytorch-cuda | 128 | 15 | 2.271 | 0 |
| scatter_default_f32_n1024 | pytorch-cuda | 256 | 15 | 3.141 | 0 |
| scatter_default_f64_n1024 | pytorch-cuda | 256 | 15 | 3.233 | 0 |
| cast_default_f32_n1024 | pytorch-cuda | 512 | 15 | 3.555 | 0 |
| cast_default_f64_n1024 | pytorch-cuda | 512 | 15 | 3.540 | 0 |
| pad_default_f32_n1048576 | pytorch-cuda | 63 | 15 | 0.857 | 0 |
| pad_default_f64_n1048576 | pytorch-cuda | 31 | 15 | 0.480 | 0 |
| slice_default_f32_n1048576 | pytorch-cuda | 128 | 15 | 1.273 | 0 |
| slice_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.651 | 0 |
| reverse_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.493 | 0 |
| reverse_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.346 | 0 |
| concatenate_default_f32_n1048576 | pytorch-cuda | 32 | 15 | 0.322 | 0 |
| concatenate_default_f64_n1048576 | pytorch-cuda | 16 | 15 | 0.412 | 0 |
| extract_diag_default_f32_n1048576 | pytorch-cuda | 512 | 15 | 3.836 | 0 |
| extract_diag_default_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.845 | 0 |
| embed_diag_default_f32_n1024 | pytorch-cuda | 64 | 15 | 0.747 | 0 |
| embed_diag_default_f64_n1024 | pytorch-cuda | 32 | 15 | 0.380 | 0 |
| tril_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.523 | 0 |
| tril_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.347 | 0 |
| triu_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.533 | 0 |
| triu_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.353 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.285 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.241 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | 64 | 15 | 1.175 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | pytorch-cuda | 64 | 15 | 1.136 | 0 |
| scatter_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.884 | 0 |
| scatter_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.728 | 0 |
| cast_default_f32_n1048576 | pytorch-cuda | 32 | 15 | 0.339 | 0 |
| cast_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.585 | 0 |
| broadcast_in_dim_f32_n1024 | pytorch-cuda | 256 | 15 | 2.019 | 0 |
| broadcast_in_dim_f64_n1024 | pytorch-cuda | 256 | 15 | 2.000 | 0 |
| convert_f32_f64_n1024 | pytorch-cuda | 512 | 15 | 3.572 | 0 |
| broadcast_in_dim_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.515 | 0 |
| broadcast_in_dim_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.266 | 0 |
| convert_f32_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.341 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_clamped_f32_n1024 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| gather_axis0_f32_n1024 | pytorch-cuda | 256 | 15 | 2.529 | 0 |
| gather_axis1_f32_n1024 | pytorch-cuda | 256 | 15 | 2.344 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_clamped_f64_n1024 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| gather_axis0_f64_n1024 | pytorch-cuda | 256 | 15 | 2.513 | 0 |
| gather_axis1_f64_n1024 | pytorch-cuda | 256 | 15 | 2.505 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_clamped_f32_n1048576 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| gather_axis0_f32_n1048576 | pytorch-cuda | 128 | 15 | 1.256 | 0 |
| gather_axis1_f32_n1048576 | pytorch-cuda | 128 | 15 | 1.287 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_clamped_f64_n1048576 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| gather_axis0_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.632 | 0 |
| gather_axis1_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.643 | 0 |
| pad_interior_f32_n1024 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_crop_f32_n1024 | pytorch-cuda | 256 | 15 | 3.196 | 0 |
| pad_interior_crop_f32_n1024 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_interior_f64_n1024 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_crop_f64_n1024 | pytorch-cuda | 256 | 15 | 3.214 | 0 |
| pad_interior_crop_f64_n1024 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_interior_f32_n1048576 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_crop_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.804 | 0 |
| pad_interior_crop_f32_n1048576 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_interior_f64_n1048576 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_crop_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.506 | 0 |
| pad_interior_crop_f64_n1048576 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| reshape_copy_f32_n1024 | pytorch-cuda | 256 | 15 | 1.970 | 0 |
| stack_axis0_f32_n1024 | pytorch-cuda | 512 | 15 | 3.153 | 0 |
| stack_axis2_f32_n1024 | pytorch-cuda | 512 | 15 | 3.622 | 0 |
| reshape_copy_f64_n1024 | pytorch-cuda | 256 | 15 | 1.746 | 0 |
| stack_axis0_f64_n1024 | pytorch-cuda | 512 | 15 | 3.205 | 0 |
| stack_axis2_f64_n1024 | pytorch-cuda | 512 | 15 | 3.307 | 0 |
| reshape_copy_c32_n1024 | pytorch-cuda | 256 | 15 | 1.846 | 0 |
| stack_axis0_c32_n1024 | pytorch-cuda | 512 | 15 | 3.747 | 0 |
| stack_axis2_c32_n1024 | pytorch-cuda | 256 | 15 | 1.991 | 0 |
| reshape_copy_c64_n1024 | pytorch-cuda | 256 | 15 | 2.225 | 0 |
| stack_axis0_c64_n1024 | pytorch-cuda | 256 | 15 | 1.893 | 0 |
| stack_axis2_c64_n1024 | pytorch-cuda | 512 | 15 | 3.739 | 0 |
| reshape_copy_i32_n1024 | pytorch-cuda | 256 | 15 | 2.041 | 0 |
| stack_axis0_i32_n1024 | pytorch-cuda | 512 | 15 | 3.705 | 0 |
| stack_axis2_i32_n1024 | pytorch-cuda | 512 | 15 | 3.853 | 0 |
| reshape_copy_i64_n1024 | pytorch-cuda | 256 | 15 | 2.174 | 0 |
| stack_axis0_i64_n1024 | pytorch-cuda | 512 | 15 | 3.909 | 0 |
| stack_axis2_i64_n1024 | pytorch-cuda | 512 | 15 | 3.397 | 0 |
| reshape_copy_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.444 | 0 |
| stack_axis0_f32_n1048576 | pytorch-cuda | 32 | 15 | 0.284 | 0 |
| stack_axis2_f32_n1048576 | pytorch-cuda | 32 | 15 | 0.828 | 0 |
| reshape_copy_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.296 | 0 |
| stack_axis0_f64_n1048576 | pytorch-cuda | 16 | 15 | 0.405 | 0 |
| stack_axis2_f64_n1048576 | pytorch-cuda | 16 | 15 | 0.734 | 0 |
| reshape_copy_c32_n1048576 | pytorch-cuda | 32 | 15 | 0.279 | 0 |
| stack_axis0_c32_n1048576 | pytorch-cuda | 16 | 15 | 0.406 | 0 |
| stack_axis2_c32_n1048576 | pytorch-cuda | 16 | 15 | 0.737 | 0 |
| reshape_copy_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.399 | 0 |
| stack_axis0_c64_n1048576 | pytorch-cuda | 8 | 15 | 0.379 | 0 |
| stack_axis2_c64_n1048576 | pytorch-cuda | 8 | 15 | 0.810 | 0 |
| reshape_copy_i32_n1048576 | pytorch-cuda | 64 | 15 | 0.443 | 0 |
| stack_axis0_i32_n1048576 | pytorch-cuda | 32 | 15 | 0.284 | 0 |
| stack_axis2_i32_n1048576 | pytorch-cuda | 32 | 15 | 0.821 | 0 |
| reshape_copy_i64_n1048576 | pytorch-cuda | 32 | 15 | 0.278 | 0 |
| stack_axis0_i64_n1048576 | pytorch-cuda | 16 | 15 | 0.404 | 0 |
| stack_axis2_i64_n1048576 | pytorch-cuda | 16 | 15 | 0.736 | 0 |
| pad_default_f32_n1024 | jax-cuda | 32 | 15 | 2.249 | 0 |
| pad_default_f64_n1024 | jax-cuda | 32 | 15 | 2.237 | 0 |
| slice_default_f32_n1024 | jax-cuda | 32 | 15 | 2.236 | 0 |
| slice_default_f64_n1024 | jax-cuda | 32 | 15 | 2.242 | 0 |
| reverse_default_f32_n1024 | jax-cuda | 64 | 15 | 3.115 | 0 |
| reverse_default_f64_n1024 | jax-cuda | 64 | 15 | 3.115 | 0 |
| concatenate_default_f32_n1024 | jax-cuda | 32 | 15 | 3.128 | 0 |
| concatenate_default_f64_n1024 | jax-cuda | 32 | 15 | 3.125 | 0 |
| extract_diag_default_f32_n1024 | jax-cuda | 64 | 15 | 3.078 | 0 |
| extract_diag_default_f64_n1024 | jax-cuda | 64 | 15 | 3.034 | 0 |
| embed_diag_default_f32_n32 | jax-cuda | 64 | 15 | 3.061 | 0 |
| embed_diag_default_f64_n32 | jax-cuda | 64 | 15 | 3.034 | 0 |
| tril_default_f32_n1024 | jax-cuda | 64 | 15 | 2.550 | 0 |
| tril_default_f64_n1024 | jax-cuda | 64 | 15 | 2.563 | 0 |
| triu_default_f32_n1024 | jax-cuda | 64 | 15 | 2.551 | 0 |
| triu_default_f64_n1024 | jax-cuda | 64 | 15 | 2.551 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | jax-cuda | 64 | 15 | 2.773 | 0 |
| take_along_axis_batched_indices_f32_n1024 | jax-cuda | 64 | 15 | 2.750 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | jax-cuda | 64 | 15 | 2.768 | 0 |
| take_along_axis_batched_indices_f64_n1024 | jax-cuda | 64 | 15 | 2.769 | 0 |
| scatter_default_f32_n1024 | jax-cuda | 32 | 15 | 3.423 | 0 |
| scatter_default_f64_n1024 | jax-cuda | 32 | 15 | 3.470 | 0 |
| cast_default_f32_n1024 | jax-cuda | 32 | 15 | 2.690 | 0 |
| cast_default_f64_n1024 | jax-cuda | 32 | 15 | 2.697 | 0 |
| pad_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.321 | 0 |
| pad_default_f64_n1048576 | jax-cuda | 31 | 15 | 2.224 | 0 |
| slice_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.237 | 0 |
| slice_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.270 | 0 |
| reverse_default_f32_n1048576 | jax-cuda | 64 | 15 | 3.155 | 0 |
| reverse_default_f64_n1048576 | jax-cuda | 32 | 15 | 1.604 | 0 |
| concatenate_default_f32_n1048576 | jax-cuda | 32 | 15 | 3.150 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda | 16 | 15 | 1.618 | 0 |
| extract_diag_default_f32_n1048576 | jax-cuda | 64 | 15 | 3.075 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda | 64 | 15 | 3.055 | 0 |
| embed_diag_default_f32_n1024 | jax-cuda | 64 | 15 | 3.150 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda | 32 | 15 | 1.588 | 0 |
| tril_default_f32_n1048576 | jax-cuda | 64 | 15 | 2.641 | 0 |
| tril_default_f64_n1048576 | jax-cuda | 32 | 15 | 1.333 | 0 |
| triu_default_f32_n1048576 | jax-cuda | 64 | 15 | 2.647 | 0 |
| triu_default_f64_n1048576 | jax-cuda | 32 | 15 | 1.373 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | jax-cuda | 64 | 15 | 2.857 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | jax-cuda | 64 | 15 | 2.857 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | 64 | 15 | 2.869 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | jax-cuda | 64 | 15 | 2.823 | 0 |
| scatter_default_f32_n1048576 | jax-cuda | 32 | 15 | 3.452 | 0 |
| scatter_default_f64_n1048576 | jax-cuda | 32 | 15 | 3.484 | 0 |
| cast_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.754 | 0 |
| cast_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.727 | 0 |
| broadcast_in_dim_f32_n1024 | jax-cuda | 32 | 15 | 2.219 | 0 |
| broadcast_in_dim_f64_n1024 | jax-cuda | 32 | 15 | 2.229 | 0 |
| convert_f32_f64_n1024 | jax-cuda | 32 | 15 | 2.692 | 0 |
| broadcast_in_dim_f32_n1048576 | jax-cuda | 32 | 15 | 2.248 | 0 |
| broadcast_in_dim_f64_n1048576 | jax-cuda | 32 | 15 | 2.281 | 0 |
| convert_f32_f64_n1048576 | jax-cuda | 32 | 15 | 2.736 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | jax-cuda | 32 | 15 | 2.814 | 0 |
| dynamic_slice_clamped_f32_n1024 | jax-cuda | 32 | 15 | 2.819 | 0 |
| gather_axis0_f32_n1024 | jax-cuda | 32 | 15 | 2.497 | 0 |
| gather_axis1_f32_n1024 | jax-cuda | 32 | 15 | 2.522 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | jax-cuda | 32 | 15 | 2.835 | 0 |
| dynamic_slice_clamped_f64_n1024 | jax-cuda | 32 | 15 | 2.838 | 0 |
| gather_axis0_f64_n1024 | jax-cuda | 32 | 15 | 2.491 | 0 |
| gather_axis1_f64_n1024 | jax-cuda | 32 | 15 | 2.514 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | jax-cuda | 32 | 15 | 2.827 | 0 |
| dynamic_slice_clamped_f32_n1048576 | jax-cuda | 32 | 15 | 2.842 | 0 |
| gather_axis0_f32_n1048576 | jax-cuda | 32 | 15 | 2.502 | 0 |
| gather_axis1_f32_n1048576 | jax-cuda | 32 | 15 | 2.533 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda | 32 | 15 | 2.836 | 0 |
| dynamic_slice_clamped_f64_n1048576 | jax-cuda | 32 | 15 | 2.838 | 0 |
| gather_axis0_f64_n1048576 | jax-cuda | 32 | 15 | 2.565 | 0 |
| gather_axis1_f64_n1048576 | jax-cuda | 32 | 15 | 2.557 | 0 |
| pad_interior_f32_n1024 | jax-cuda | 32 | 15 | 2.280 | 0 |
| pad_crop_f32_n1024 | jax-cuda | 32 | 15 | 2.274 | 0 |
| pad_interior_crop_f32_n1024 | jax-cuda | 32 | 15 | 2.292 | 0 |
| pad_interior_f64_n1024 | jax-cuda | 32 | 15 | 2.294 | 0 |
| pad_crop_f64_n1024 | jax-cuda | 32 | 15 | 2.285 | 0 |
| pad_interior_crop_f64_n1024 | jax-cuda | 32 | 15 | 2.298 | 0 |
| pad_interior_f32_n1048576 | jax-cuda | 10 | 15 | 0.779 | 0 |
| pad_crop_f32_n1048576 | jax-cuda | 32 | 15 | 2.324 | 0 |
| pad_interior_crop_f32_n1048576 | jax-cuda | 10 | 15 | 0.779 | 0 |
| pad_interior_f64_n1048576 | jax-cuda | 5 | 15 | 0.436 | 0 |
| pad_crop_f64_n1048576 | jax-cuda | 32 | 15 | 2.324 | 0 |
| pad_interior_crop_f64_n1048576 | jax-cuda | 5 | 15 | 0.435 | 0 |
| reshape_copy_f32_n1024 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_f32_n1024 | jax-cuda | 32 | 15 | 2.610 | 0 |
| stack_axis2_f32_n1024 | jax-cuda | 32 | 15 | 2.623 | 0 |
| reshape_copy_f64_n1024 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_f64_n1024 | jax-cuda | 32 | 15 | 2.626 | 0 |
| stack_axis2_f64_n1024 | jax-cuda | 32 | 15 | 2.693 | 0 |
| reshape_copy_c32_n1024 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_c32_n1024 | jax-cuda | 32 | 15 | 3.134 | 0 |
| stack_axis2_c32_n1024 | jax-cuda | 32 | 15 | 3.135 | 0 |
| reshape_copy_c64_n1024 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_c64_n1024 | jax-cuda | 32 | 15 | 2.855 | 0 |
| stack_axis2_c64_n1024 | jax-cuda | 32 | 15 | 3.140 | 0 |
| reshape_copy_i32_n1024 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_i32_n1024 | jax-cuda | 32 | 15 | 2.691 | 0 |
| stack_axis2_i32_n1024 | jax-cuda | 32 | 15 | 2.820 | 0 |
| reshape_copy_i64_n1024 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_i64_n1024 | jax-cuda | 32 | 15 | 2.831 | 0 |
| stack_axis2_i64_n1024 | jax-cuda | 32 | 15 | 3.094 | 0 |
| reshape_copy_f32_n1048576 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_f32_n1048576 | jax-cuda | 32 | 15 | 2.646 | 0 |
| stack_axis2_f32_n1048576 | jax-cuda | 32 | 15 | 3.141 | 0 |
| reshape_copy_f64_n1048576 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_f64_n1048576 | jax-cuda | 16 | 15 | 1.374 | 0 |
| stack_axis2_f64_n1048576 | jax-cuda | 16 | 15 | 1.613 | 0 |
| reshape_copy_c32_n1048576 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_c32_n1048576 | jax-cuda | 16 | 15 | 1.619 | 0 |
| stack_axis2_c32_n1048576 | jax-cuda | 16 | 15 | 1.620 | 0 |
| reshape_copy_c64_n1048576 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_c64_n1048576 | jax-cuda | 8 | 15 | 0.791 | 0 |
| stack_axis2_c64_n1048576 | jax-cuda | 8 | 15 | 0.875 | 0 |
| reshape_copy_i32_n1048576 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_i32_n1048576 | jax-cuda | 32 | 15 | 3.138 | 0 |
| stack_axis2_i32_n1048576 | jax-cuda | 32 | 15 | 3.170 | 0 |
| reshape_copy_i64_n1048576 | jax-cuda | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_i64_n1048576 | jax-cuda | 16 | 15 | 1.605 | 0 |
| stack_axis2_i64_n1048576 | jax-cuda | 16 | 15 | 1.604 | 0 |
| pad_default_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.590 | 0 |
| pad_default_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.597 | 0 |
| slice_default_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.597 | 0 |
| slice_default_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.586 | 0 |
| reverse_default_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.587 | 0 |
| reverse_default_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.572 | 0 |
| concatenate_default_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.723 | 0 |
| concatenate_default_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.725 | 0 |
| extract_diag_default_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.584 | 0 |
| extract_diag_default_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.544 | 0 |
| embed_diag_default_f32_n32 | jax-cuda-jit | 64 | 15 | 2.569 | 0 |
| embed_diag_default_f64_n32 | jax-cuda-jit | 64 | 15 | 2.541 | 0 |
| tril_default_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.529 | 0 |
| tril_default_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.564 | 0 |
| triu_default_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.545 | 0 |
| triu_default_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.535 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.730 | 0 |
| take_along_axis_batched_indices_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.717 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.686 | 0 |
| take_along_axis_batched_indices_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.707 | 0 |
| scatter_default_f32_n1024 | jax-cuda-jit | 64 | 15 | 3.428 | 0 |
| scatter_default_f64_n1024 | jax-cuda-jit | 64 | 15 | 3.439 | 0 |
| cast_default_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.572 | 0 |
| cast_default_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.583 | 0 |
| pad_default_f32_n1048576 | jax-cuda-jit | 63 | 15 | 2.627 | 0 |
| pad_default_f64_n1048576 | jax-cuda-jit | 31 | 15 | 1.301 | 0 |
| slice_default_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.660 | 0 |
| slice_default_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.653 | 0 |
| reverse_default_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.653 | 0 |
| reverse_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.351 | 0 |
| concatenate_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 1.437 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda-jit | 16 | 15 | 0.716 | 0 |
| extract_diag_default_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.577 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.566 | 0 |
| embed_diag_default_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.659 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 1.330 | 0 |
| tril_default_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.662 | 0 |
| tril_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.325 | 0 |
| triu_default_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.680 | 0 |
| triu_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.353 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.835 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.815 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.839 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.844 | 0 |
| scatter_default_f32_n1048576 | jax-cuda-jit | 64 | 15 | 3.518 | 0 |
| scatter_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.769 | 0 |
| cast_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 1.345 | 0 |
| cast_default_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.663 | 0 |
| broadcast_in_dim_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.556 | 0 |
| broadcast_in_dim_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.568 | 0 |
| convert_f32_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.559 | 0 |
| broadcast_in_dim_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.653 | 0 |
| broadcast_in_dim_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.346 | 0 |
| convert_f32_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.343 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.818 | 0 |
| dynamic_slice_clamped_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.800 | 0 |
| gather_axis0_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.732 | 0 |
| gather_axis1_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.747 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.802 | 0 |
| dynamic_slice_clamped_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.798 | 0 |
| gather_axis0_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.758 | 0 |
| gather_axis1_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.760 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.866 | 0 |
| dynamic_slice_clamped_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.849 | 0 |
| gather_axis0_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.822 | 0 |
| gather_axis1_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.840 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.874 | 0 |
| dynamic_slice_clamped_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.904 | 0 |
| gather_axis0_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.857 | 0 |
| gather_axis1_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.826 | 0 |
| pad_interior_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.565 | 0 |
| pad_crop_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.560 | 0 |
| pad_interior_crop_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.558 | 0 |
| pad_interior_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.563 | 0 |
| pad_crop_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.542 | 0 |
| pad_interior_crop_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.539 | 0 |
| pad_interior_f32_n1048576 | jax-cuda-jit | 10 | 15 | 0.445 | 0 |
| pad_crop_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.258 | 0 |
| pad_interior_crop_f32_n1048576 | jax-cuda-jit | 10 | 15 | 0.363 | 0 |
| pad_interior_f64_n1048576 | jax-cuda-jit | 5 | 15 | 0.304 | 0 |
| pad_crop_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.140 | 0 |
| pad_interior_crop_f64_n1048576 | jax-cuda-jit | 5 | 15 | 0.350 | 0 |
| reshape_copy_f32_n1024 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.311 | 0 |
| stack_axis2_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.699 | 0 |
| reshape_copy_f64_n1024 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.689 | 0 |
| stack_axis2_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.411 | 0 |
| reshape_copy_c32_n1024 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_c32_n1024 | jax-cuda-jit | 64 | 15 | 2.343 | 0 |
| stack_axis2_c32_n1024 | jax-cuda-jit | 64 | 15 | 2.317 | 0 |
| reshape_copy_c64_n1024 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_c64_n1024 | jax-cuda-jit | 64 | 15 | 2.720 | 0 |
| stack_axis2_c64_n1024 | jax-cuda-jit | 64 | 15 | 2.744 | 0 |
| reshape_copy_i32_n1024 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_i32_n1024 | jax-cuda-jit | 64 | 15 | 2.548 | 0 |
| stack_axis2_i32_n1024 | jax-cuda-jit | 64 | 15 | 2.740 | 0 |
| reshape_copy_i64_n1024 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_i64_n1024 | jax-cuda-jit | 64 | 15 | 2.723 | 0 |
| stack_axis2_i64_n1024 | jax-cuda-jit | 64 | 15 | 2.729 | 0 |
| reshape_copy_f32_n1048576 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_f32_n1048576 | jax-cuda-jit | 32 | 15 | 1.425 | 0 |
| stack_axis2_f32_n1048576 | jax-cuda-jit | 32 | 15 | 1.425 | 0 |
| reshape_copy_f64_n1048576 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_f64_n1048576 | jax-cuda-jit | 16 | 15 | 0.665 | 0 |
| stack_axis2_f64_n1048576 | jax-cuda-jit | 16 | 15 | 0.696 | 0 |
| reshape_copy_c32_n1048576 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_c32_n1048576 | jax-cuda-jit | 16 | 15 | 0.604 | 0 |
| stack_axis2_c32_n1048576 | jax-cuda-jit | 16 | 15 | 0.723 | 0 |
| reshape_copy_c64_n1048576 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_c64_n1048576 | jax-cuda-jit | 8 | 15 | 0.460 | 0 |
| stack_axis2_c64_n1048576 | jax-cuda-jit | 8 | 15 | 0.522 | 0 |
| reshape_copy_i32_n1048576 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_i32_n1048576 | jax-cuda-jit | 32 | 15 | 1.578 | 0 |
| stack_axis2_i32_n1048576 | jax-cuda-jit | 32 | 15 | 1.561 | 0 |
| reshape_copy_i64_n1048576 | jax-cuda-jit | — | — | — | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| stack_axis0_i64_n1048576 | jax-cuda-jit | 16 | 15 | 0.752 | 0 |
| stack_axis2_i64_n1048576 | jax-cuda-jit | 16 | 15 | 0.746 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/structural
suite_file: benchmarks/gpu/structural.yaml
timestamp: '2026-11-02T21:06:08Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 433121a7756fdb4f109b02f0422d3714a04514c6
  dirty: false
  features:
  - cuda
  resolved_path: /home/shinaoka/projects/tensor4all/tenferro-batched-gemm
harness:
  path: /workspaces/tenferro-benchmark
  commit: 8987033651c18a03e47e16cdd365df807b77b6be
  dirty: true
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
  hostname: 0fe295fad79b
  os: Linux-6.8.0-147-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-7
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
  device_name: NVIDIA A100 80GB PCIe
  device_uuid: GPU-530977e1-4968-9283-4129-9fbec3e66542
  device_memory_bytes: 85899345920
  driver_version: 580.178.04
  runtime_version: '12.9'
  cuda_version: '13.0'
  cudnn_version: '92700'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-530977e1-4968-9283-4129-9fbec3e66542
  cuda_visible_devices: GPU-530977e1-4968-9283-4129-9fbec3e66542
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh structural
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 118
  selected_cases: 118
  sha256:
    .devcontainer/cuda/devcontainer.json: 0bf1465bbd1a4c9f896d5320838f2d5a488632a59af7f9b59cc050e95e0b8e09
    .devcontainer/cuda/Dockerfile: 578d9815e5145aa1651ac7e93833468c345ee8bf5e95bfa7717551580ae9feca
    src/bin/benchmark_gpu_public_api.rs: 1a5f3469fc9ffde64a79e05eb9853632366855ece3280ba83e2ba21fe3fb2cdc
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 9c3d7916a3281d2dfd1caa68551dc9c81c79dab6b7525324339e5c4816581e6d
    src/bin/benchmark_gpu_public_api/multi.rs: 291e63a517b6f7108527254a7f83751420aa1d4e73395aa60f0691312b2a3773
    src/bin/benchmark_gpu_public_api/plan.rs: 2d87b9d73bf25df8797133cf0a8dc8b1fa974fc3bee055ad92e42802c9ebf643
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: a5a4aa0edb0a1158f6669b642f4bc25a7079314fc2a711ef3ef57cc19c0a35b2
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: c9d61b1e99bda517a59a6e34f625fd4d651bbbb19f6dc06ba4d0d1c0d50dfd6a
    src/bin/benchmark_gpu_public_api/state.rs: e11a863aee7a85822c474eb771b45ec402e21dbe92173d178c42119ab8ca307e
    src/bin/benchmark_gpu_public_api/structural.rs: 8b61ac01b58edc090eafbc124ec135d5efeb4979631a939656ba78dd6e5c042a
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: ddc112ea5d568a0399378f2be8b31c91b3144fe7d286e8094d27ca5ccda959bf
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 9cbcb473cba55b1605bab07fd3ff14247859fd7fe9e73d5378dd70108313256b
    data/results/nvidia-gpu/gpu/structural/20261102_210608/fixtures/manifest.json: 6b4532ecb67f54e60c796185be6597f7f29b0ca79754f79e16488a51241e9220
    target/release/benchmark_gpu_public_api: c0a0be40dd246cd4cb7ca59cac19bacd1549a4f3a593aea91e887cf41f6e4aa6
  reference_numpy_version: 2.4.2
```
