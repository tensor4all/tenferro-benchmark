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
helpers="$(mktemp -d)"
trap 'rm -rf "$helpers"' EXIT
# These reviewed CI helpers are pinned independently of the compiled test source.
helper_ref=f3b995657adbbcabc636c5377877205d920a35aa
for helper in install_cuda_runtime_tree.sh seed_cuda_runtime_tree.sh; do
  curl -fsSL --retry 3 -o "$helpers/$helper" "https://raw.githubusercontent.com/tensor4all/tenferro-rs/$helper_ref/scripts/ci/$helper"
done
bash "$helpers/install_cuda_runtime_tree.sh" "$CUDA_RUNTIME_VERSION" "$payload/usr/local/cuda-$CUDA_RUNTIME_VERSION"
python3 scripts/ci/check_cuda_headers.py --cuda-root "$payload/usr/local/cuda-$CUDA_RUNTIME_VERSION"
tar --zstd -cf tenferro-rs/gpu-ci-runtime.tar.zst -C "$payload" .
