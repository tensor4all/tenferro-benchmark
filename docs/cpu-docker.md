# CPU benchmark Docker image

`docker/cpu/Dockerfile` builds the single CPU benchmark image (Ubuntu 24.04,
Rust stable, Python 3 + venv, OpenBLAS, MKL, AOCL 6.0.0, PyTorch CPU, JAX CPU).
Source repos are mounted at `/work`, not baked in. Default command is `/bin/bash`
(`CMD`, not `ENTRYPOINT`, so `docker run <image> bash -lc '...'` works).

Published image: `shinaoka/tenferro-bench-cpu:cpu-20261120`
(`sha256:739c9caac797ba944b611581ebd3a9c6d82575a55d56bdbcbba2a2e3eade5914`).

Build: `docker build -t shinaoka/tenferro-bench-cpu:cpu-20261120 docker/cpu`

Pin only the measured container, never `dockerd`:

```
docker run --rm --cpuset-cpus=4-11 -v <tensor4all>:/work -w /work/colmat-rs <image> \
  bash -lc 'cargo test -p colmat --lib --release'
```

Smoke only (not a benchmark run); results recorded in issue #502.
