FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV OPENBLAS_ROOT=/opt/openblas
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
ENV PATH="/home/vscode/.cargo/bin:/home/vscode/.local/bin:${PATH}"

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
        libopenblas-dev \
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

RUN set -eux; \
    mkdir -p /opt/openblas/include /opt/openblas/lib; \
    cblas_header="$(find /usr/include -name cblas.h | head -n 1)"; \
    test -n "$cblas_header"; \
    cp -a "$(dirname "$cblas_header")"/. /opt/openblas/include/; \
    libopenblas="$(find /usr/lib \( -name 'libopenblas.so' -o -name 'libopenblas.so.*' \) | sort | head -n 1)"; \
    test -n "$libopenblas"; \
    ln -sf "$libopenblas" /opt/openblas/lib/libopenblas.so

USER vscode

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable \
    && rustup component add clippy rustfmt \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

USER root

RUN mkdir -p /home/vscode/.cargo/registry /home/vscode/.cargo/git /home/vscode/.cache/uv \
    && chown -R vscode:vscode /home/vscode/.cargo /home/vscode/.cache
