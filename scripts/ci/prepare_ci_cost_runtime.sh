#!/usr/bin/env bash
# Hosted preparation for the slim-image diagnostic; never build on the GPU.
set -euo pipefail
payload="$PWD/runtime-payload"
mkdir -p "$payload/bin" "$payload/wheels" "$payload/opt/tenferro-ci" "$payload/usr/local"
cp "$(rustup which cargo)" "$payload/bin/cargo"
cp "$(command -v cargo-nextest)" "$payload/bin/cargo-nextest"
bash tenferro-rs/scripts/ci/install_cutensor.sh "$CUTENSOR_VERSION" "$payload/opt/tenferro-ci/cutensor-$CUTENSOR_VERSION"
python3 -m pip download --only-binary=:all: --no-deps --python-version 312 --platform manylinux_2_27_x86_64 --platform manylinux2014_x86_64 --implementation cp --abi cp312 \
  --dest "$payload/wheels" "jax-cuda12-pjrt==$JAX_CUDA12_PJRT_VERSION" \
  "nvidia-cudnn-cu12==$NVIDIA_CUDNN_CU12_VERSION" "nvidia-cuda-nvcc-cu12==$NVIDIA_CUDA_NVCC_CU12_VERSION"
headers="$(mktemp -d)"
trap 'rm -rf "$headers"' EXIT
for pkg in cuda-crt-12-8_12.8.93-1_amd64.deb cuda-cudart-dev-12-8_12.8.90-1_amd64.deb cuda-cccl-12-8_12.8.90-1_amd64.deb cuda-driver-dev-12-8_12.8.90-1_amd64.deb; do
  curl -fsSL --retry 3 -o "$headers/$pkg" "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/$pkg"
  dpkg-deb -x "$headers/$pkg" "$headers/tree"
done
cp -a "$headers/tree/usr/local/cuda-12.8" "$payload/usr/local/"
tar --zstd -cf tenferro-rs/gpu-ci-runtime.tar.zst -C "$payload" .
