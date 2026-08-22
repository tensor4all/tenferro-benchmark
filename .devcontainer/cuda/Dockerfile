FROM nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV OPENBLAS_ROOT=/opt/openblas
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/home/vscode/.cargo/bin:/home/vscode/.local/bin:${PATH}"
ENV LD_LIBRARY_PATH="${OPENBLAS_ROOT}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

ARG OPENBLAS_VERSION=0.3.26
ARG OPENBLAS_MAX_THREADS=64
ARG OPENBLAS_DYNAMIC_ARCH=1
ARG CUTLASS_TAG=v3.7.0
ARG GINKGO_TAG=v1.8.0
# Docker builds do not have GPU access, so make the target architecture explicit.
# 86 matches the RTX 30-series benchmark hosts; override this build arg for a
# different NVIDIA architecture.
ARG GINKGO_CUDA_ARCHITECTURES=86

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash-completion \
        build-essential \
        ca-certificates \
        cmake \
        ccache \
        curl \
        file \
        gdb \
        gfortran \
        git \
        lld \
        libcutensor2 \
        libcutensor-dev \
        libffi-dev \
        libomp-dev \
        libssl-dev \
        ninja-build \
        patchelf \
        pkg-config \
        python3-pip \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        ripgrep \
        sudo \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=vscode

# The NVIDIA Ubuntu 24.04 base image ships a user named 'ubuntu' at UID/GID 1000.
# Rename it to 'vscode' to match the standard devcontainer convention.
RUN usermod  -l ${USERNAME} ubuntu \
    && groupmod -n ${USERNAME} ubuntu \
    && usermod  -d /home/${USERNAME} -m ${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

COPY .devcontainer/install_openblas.sh /usr/local/bin/install_openblas.sh
COPY scripts/setup_gpu_vendors.sh /usr/local/bin/setup_gpu_vendors.sh

RUN bash /usr/local/bin/install_openblas.sh

# Bake the optional comparison backends into the image.  Keeping them under
# /opt makes them independent of the workspace mount and ready on first start.
RUN CUTLASS_DIR=/opt/cutlass \
    CUTLASS_TAG="${CUTLASS_TAG}" \
    GINKGO_DIR=/opt/ginkgo \
    GINKGO_SRC=/tmp/ginkgo-src \
    GINKGO_TAG="${GINKGO_TAG}" \
    CUDA_ARCH="${GINKGO_CUDA_ARCHITECTURES}" \
    bash /usr/local/bin/setup_gpu_vendors.sh all \
    && rm -rf /tmp/ginkgo-src

USER vscode

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable \
    && rustup component add clippy rustfmt \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

USER root

RUN mkdir -p /home/vscode/.cargo/bin \
    && chown -R vscode:vscode /home/vscode/.cargo
