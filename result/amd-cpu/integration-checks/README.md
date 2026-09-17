# CPU optimization integration checks (2026-09-17)

These are correctness/build logs, not new timing measurements.

- strided-rs PR259 merged as `5bc5ab75a20277f0c8820cb288b23f6bb6dfbd91`.
- tenferro-rs PR1807 source `6ac70cd07e802dee3bda969bc40e0ffd91f37168`;
  its follow-up `c184115e` updates only the CI test's expected dependency hash.
- Docker image `sha256:228bf2e4baf4ef4117a65bb8a5a2d58247ce0c0985aec8a8fb53b7a58193893f`,
  Rust1.98.1, shared OpenBLAS0.3.26. All six strided crates resolve from GitHub,
  without Cargo path/manifest overrides. Runtime OpenBLAS/OMP/Rayon threads1;
  Cargo jobs16 and Rust test harness threads1.

`tenferro-pr-fast.log.gz`: passed formatting, doc snippets, strict workspace
Clippy plus tropical/sparse/TBLIS extension Clippy, and 1439 tests/doctests:

```sh
bash scripts/check-pr-fast.sh --base origin/main --coverage-reviewed \
  --test 'cargo test -j 16 -p tenferro-cpu -p tenferro-ad --no-fail-fast -- --test-threads=1'
```

`tenferro-pr-blas.log.gz`: six BLAS-only tests and one injected-provider opt-out:

```sh
export RUSTFLAGS='-L native=/opt/openblas/lib -l openblas'
cargo test -j 16 -p tenferro-cpu --no-default-features --features cpu-blas \
  --lib blas_uninit_tests -- --test-threads=1
cargo test -j 16 -p tenferro-cpu --no-default-features \
  --features cpu-blas,provider-inject --lib injected_blas_has_no_uninit_witness \
  -- --test-threads=1
```

`tenferro-ci-config.log.gz`: `python3 scripts/ci/run_profile.py ci-config`
passed all328 Python checks and actionlint after updating the expected pin.
Provisioning messages in these unit tests are mocked; no cloud resources were
provisioned by this local check.

These gates do not establish native performance acceptance or the deferred4T
matrix. The benchmark's full timed linalg-AD integration script was not rerun.
