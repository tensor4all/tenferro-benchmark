# tprims provider comparison

`--features tprims` builds every tenferro CPU backend of this suite with the
tprims GEMM and general-contraction providers from
`extern/tenferro-rs/ext/tenferro-cpu-tprims` installed
([tensor4all/tprims-rs](https://github.com/tensor4all/tprims-rs) phase 1e;
tenferro-rs #1953). Everything tprims does not handle falls back to the
backend's own kind. The acceptance comparison is a default build against a
`--features tprims` build of the same commits, with the paired ABBA runner.

The feature needs an `extern/tenferro-rs` that contains
`ext/tenferro-cpu-tprims` (tenferro-rs #1955 or later): like every path
dependency, its manifest is resolved even when the feature is off.

CPU linear algebra is not a tprims route. tenferro runs every CPU linalg
family through the extracted `tlinalg` (faer) and `tlinalg-blas` (LAPACK)
providers (tenferro-rs #1956), and the ext crate no longer ships linalg
kernels; linalg rows measure the same code in a default and a
`--features tprims` build.

`TPRIMS_ROUTES` (comma-separated `gemm`, `contract`; default both) selects
which slots get tprims, so each operation family is accepted or rejected on
its own. With `contract` installed every `dot_general` goes
through tprims-contract; `TPRIMS_ROUTES=gemm` lets tenferro lower
contractions itself and hand the GEMMs to tprims, which is also how
batched-GEMM shapes are logged. Binaries print
`TENFERRO_CPU_PROVIDER=default|tprims(<routes>)`.

## Measuring

Follow the `tprims-benchmark` skill of tprims-rs for host preparation: idle
cores of one L3 domain (`benchmarks/scripts/idle_cpus.py pick N`), every
measurement pinned and checked idle before and after
(`benchmarks/scripts/pinned.sh`), runs sequential, build jobs from
`CARGO_BUILD_JOBS`. Thread counts 1, 4 and 8 decide acceptance; 16 is
reported only.

## Shape profile (corpus for tprims-rs)

With the feature, `TPRIMS_SHAPE_LOG=<file>` appends one JSON line per
provider call (operation, dtype, operand shapes and strides, contraction
axes, elapsed nanoseconds, outcome). Build a tprims-rs corpus from one or
more logs:

```bash
TPRIMS_SHAPE_LOG=/tmp/shapes.jsonl ./target/release/benchmark_cpu_public_api --output /tmp/out.csv --num-threads 1
python3 scripts/tprims_shape_corpus.py /tmp/shapes.jsonl -o corpus.json \
  --source '{"suite": "cpu/public_api", "tenferro_benchmark": "<commit>", "tenferro_rs": "<commit>"}'
```

The corpus keeps the most expensive shape groups up to `--time-share` (0.95)
of the logged time plus the `--frequent` (20) most frequent others, and is
replayed in tprims-rs with `contract --corpus` / `blas --corpus`.
