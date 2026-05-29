FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV OPENBLAS_ROOT=/opt/openblas
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/home/vscode/.cargo/bin:/home/vscode/.local/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

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
