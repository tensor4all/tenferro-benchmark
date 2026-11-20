FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV AOCL_ROOT=/opt/AMD/aocl/6.0.0/gcc/MT
ENV MKLROOT=/opt/intel/oneapi/mkl/latest
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/latest/lib:/opt/AMD/aocl/6.0.0/gcc/MT/lib
ENV PATH=/root/.cargo/bin:/opt/AMD/aocl/6.0.0/gcc/MT/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config git curl wget ca-certificates gnupg \
    python3 python3-pip python3-venv \
    libopenblas-dev liblapack-dev libomp-dev libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Intel oneAPI MKL
RUN wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      | gpg --dearmor > /usr/share/keyrings/intel-oneapi.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/intel-oneapi.gpg] https://apt.repos.intel.com/oneapi all main' \
      > /etc/apt/sources.list.d/intel-oneapi.list \
    && apt-get update && apt-get install -y --no-install-recommends intel-oneapi-mkl-devel \
    && rm -rf /var/lib/apt/lists/*

# AMD AOCL 6.0.0
RUN mkdir -p /opt/AMD/aocl \
    && wget -q https://download.amd.com/developer/eula/aocl/aocl-6-0/aocl-linux-gcc-6.0.0.tar.gz -O /tmp/aocl.tgz \
    && tar xzf /tmp/aocl.tgz -C /tmp \
    && cd /tmp/aocl-linux-gcc-6.0.0 && ./install.sh -t /opt/AMD/aocl -l blis libflame utils -i lp64 \
    && rm -rf /tmp/aocl* 

# Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable --profile minimal
RUN cargo --version && rustc --version

# Python reference arms
RUN python3 -m pip install --break-system-packages --no-cache-dir \
      numpy scipy \
    && python3 -m pip install --break-system-packages --no-cache-dir \
      torch --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install --break-system-packages --no-cache-dir 'jax[cpu]'

WORKDIR /work
CMD ["/bin/bash"]
