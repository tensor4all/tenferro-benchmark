FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV OPENBLAS_ROOT=/opt/openblas
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
ENV PATH="/home/vscode/.cargo/bin:/home/vscode/.local/bin:${PATH}"
ENV LD_LIBRARY_PATH="${OPENBLAS_ROOT}/lib"

ARG OPENBLAS_VERSION=0.3.26
ARG OPENBLAS_MAX_THREADS=64
ARG OPENBLAS_DYNAMIC_ARCH=1

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
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY .devcontainer/install_openblas.sh /usr/local/bin/install_openblas.sh

RUN bash /usr/local/bin/install_openblas.sh

USER vscode

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable \
    && rustup component add clippy rustfmt \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

USER root

RUN mkdir -p /home/vscode/.cargo/registry /home/vscode/.cargo/git /home/vscode/.cache/uv \
    && chown -R vscode:vscode /home/vscode/.cargo /home/vscode/.cache
