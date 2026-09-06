## Measurement notes

Collected in a newly built Linux CPU devcontainer on the shared AMD EPYC host.
The 1-thread collection completed before the 4-thread collection began.
No CPU selection, affinity settings, idle-host gate, or performance acceptance
threshold was added. Other host activity was not controlled; no cause is
assigned to the observed variation.

All 154 cases passed numerical checks at each thread count. There are 51/154
NOISY rows at 1 thread and 117/154 at 4 threads (CoV > 10%); maximum CoV is
83.4% and 120.6%, respectively. These measurements and all 4,620 raw timing
samples are retained as collected. Do not infer speedups or thread scaling
from these noisy rows. No rerun was requested merely to obtain quieter data.

The measured benchmark revision is `d5faef9` and tenferro-rs is `181cbadf`.
The later report-only commit adds these notes and the generated results;
it does not change the case execution or timing code.

Collection command, run from the benchmark worktree on the Linux host:

```bash
npx --yes @devcontainers/cli exec --workspace-folder . \
  --remote-env BENCHMARK_COMMIT="$(git rev-parse HEAD)" bash -lc '
  export CARGO_HOME=/tmp/benchmark95-cargo UV_CACHE_DIR=/tmp/benchmark95-uv
  TENFERRO_CPU_FEATURES=system-mkl TENFERRO_CPU_BACKEND_KIND=blas \
  BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_small_work.sh 1 4'
```

The temporary cache paths avoid permission conflicts in existing shared Docker
cache volumes; those shared volumes were not modified to fix permissions.
