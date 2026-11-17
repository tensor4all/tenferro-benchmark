FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
ENV PATH="/home/vscode/.cargo/bin:/home/vscode/.local/bin:/home/vscode/.juliaup/bin:${PATH}"

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
        gnupg \
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

USER vscode

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable \
    && rustup component add clippy rustfmt \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Julia (via juliaup) is required by the cpu/permutation suite's Julia runners
# (julia-base / strided-jl participants in scripts/benchmark_permutation.jl).
RUN curl -fsSL https://install.julialang.org \
        | sh -s -- --yes --default-channel release

USER root

RUN mkdir -p /home/vscode/.cargo/registry /home/vscode/.cargo/git /home/vscode/.cache/uv \
    && chown -R vscode:vscode /home/vscode/.cargo /home/vscode/.cache
