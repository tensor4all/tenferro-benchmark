# Build from the repository root:
# docker build -f .devcontainer/openblas/Dockerfile -t tenferro-benchmark-openblas .
FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04@sha256:d94c97dd9cacf183d0a6fd12a8e87b526e9e928307674ae9c94139139c0c6eae AS base

ENV DEBIAN_FRONTEND=noninteractive \
    OPENBLAS_ROOT=/opt/openblas \
    CMAKE_POLICY_VERSION_MINIMUM=3.5
ENV LD_LIBRARY_PATH="${OPENBLAS_ROOT}/lib"

ARG BUILD_JOBS=16
ARG OPENBLAS_VERSION=0.3.26
ARG OPENBLAS_MAX_THREADS=64
ARG OPENBLAS_DYNAMIC_ARCH=1
# Portable common code for AMD/Intel x86_64; optimized kernels dispatch at runtime.
ARG OPENBLAS_TARGET=CORE2

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl gfortran git lld \
        libffi-dev libomp-dev libssl-dev ninja-build pkg-config \
        python3 python3.12 python3.12-dev python3.12-venv ripgrep zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY .devcontainer/install_openblas.sh /usr/local/bin/install_openblas.sh
RUN bash /usr/local/bin/install_openblas.sh

FROM base AS pytorch-builder
# v2.12.0; pin the source rather than following a moving branch/tag.
ARG PYTORCH_COMMIT=0d62256a2b23365f8e1604297eb23a6545102aa8
ARG PYTORCH_BUILD_VERSION=2.12.0+openblas
ARG BUILD_JOBS=16
ENV OpenBLAS_HOME=/opt/openblas \
    BLAS=OpenBLAS \
    USE_MKL=0 USE_MKLDNN=0 \
    USE_CUDA=0 USE_ROCM=0 USE_MPS=0 USE_DISTRIBUTED=0 \
    USE_KINETO=0 BUILD_TEST=0 \
    PYTORCH_BUILD_NUMBER=1

RUN python3.12 -m venv /opt/torch-build
ENV PATH="/opt/torch-build/bin:${PATH}"
# Keep the source and compiler artifacts out of the final image. The wheel is
# retained so runner venv resets never substitute a provider-mismatched wheel.
RUN git init /tmp/pytorch \
    && cd /tmp/pytorch \
    && git remote add origin https://github.com/pytorch/pytorch.git \
    && git fetch --depth 1 origin "${PYTORCH_COMMIT}" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${PYTORCH_COMMIT}" \
    && git submodule update --init --recursive --depth 1 --jobs "${BUILD_JOBS}" \
    && python -m pip install --no-cache-dir -r requirements-build.txt \
    && MAX_JOBS="${BUILD_JOBS}" PYTORCH_BUILD_VERSION="${PYTORCH_BUILD_VERSION}" \
       python -m pip wheel --no-build-isolation --no-deps . -w /opt/pytorch-wheels \
    && cp build/CMakeCache.txt /opt/pytorch-wheels/CMakeCache.txt \
    && git rev-parse HEAD > /opt/pytorch-wheels/pytorch-commit.txt \
    && python -m pip freeze > /opt/pytorch-wheels/build-requirements.txt \
    && cd /opt/pytorch-wheels && sha256sum ./*.whl > SHA256SUMS \
    && rm -rf /tmp/pytorch

FROM base AS benchmark
COPY --from=pytorch-builder /opt/pytorch-wheels /opt/pytorch-wheels
COPY .devcontainer/openblas/verify_provider.py /usr/local/bin/verify-openblas-pytorch

ENV TENFERRO_CPU_FEATURES=system-openblas \
    PUBLICATION_GATE_FEATURES=system-openblas \
    TENFERRO_CPU_BACKEND_KIND=blas \
    BENCHMARK_TARGET_PROFILE=amd-cpu \
    BENCHMARK_TORCH_WHEEL=/opt/pytorch-wheels/torch.whl \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
    CARGO_BUILD_JOBS=16

# Keep the PEP 427 wheel filename; the runner resolves this symlink before uv.
RUN ln -s "$(find /opt/pytorch-wheels -maxdepth 1 -name 'torch-*.whl')" \
        /opt/pytorch-wheels/torch.whl \
    && python3.12 -m venv /opt/torch-check \
    && /opt/torch-check/bin/python -m pip install --no-cache-dir \
        "$(readlink -f /opt/pytorch-wheels/torch.whl)" \
    && /opt/torch-check/bin/python /usr/local/bin/verify-openblas-pytorch --threads 1 \
    && rm -rf /opt/torch-check

ENV PATH="/home/vscode/.cargo/bin:/home/vscode/.local/bin:/home/vscode/.juliaup/bin:${PATH}"
USER vscode
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable \
    && rustup component add clippy rustfmt \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
RUN curl -fsSL https://install.julialang.org \
        | sh -s -- --yes --default-channel release
USER root
RUN mkdir -p /home/vscode/.cargo/registry /home/vscode/.cargo/git /home/vscode/.cache/uv \
    && chown -R vscode:vscode /home/vscode/.cargo /home/vscode/.cache

# Protect uv invocations outside the per-runner setup helper too.
ENV UV_NO_SYNC=1
