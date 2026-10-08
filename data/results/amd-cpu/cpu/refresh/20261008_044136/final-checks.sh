#!/usr/bin/env bash
set -euo pipefail
export PATH="$PWD/.venv/bin:$PATH"
export CARGO_TARGET_DIR="$PWD/target/compatibility-mkl"
export CARGO_BUILD_JOBS=8
export LD_LIBRARY_PATH="$MKLROOT/lib:/opt/intel/oneapi/compiler/2026.1/lib:${LD_LIBRARY_PATH:-}"
rustfmt --check --edition 2021 src/main.rs src/bin/publication_gate.rs
cargo test --no-default-features --features system-mkl --all-targets
python -m unittest discover -s tests -p 'test_*.py'
python -m py_compile scripts/benchmark_python.py scripts/benchmark_cpu_ops_python.py scripts/format_cpu_ops_results.py
python scripts/validate_benchmark_suite.py benchmarks/cpu/einsum.yaml benchmarks/cpu/public_api.yaml benchmarks/gpu/dense.yaml benchmarks/gpu/einsum.yaml benchmarks/gpu/sparse.yaml benchmarks/gpu/linalg_jvp_vjp.yaml
for test in tests/test_build_for_tenferro_rev.sh tests/test_cpu_blas_provider_policy.sh tests/test_run_all_rust_bin_selection.sh tests/test_suite_result_layout.sh tests/test_run_all_docs_outputs.sh tests/test_cpu_ops_linalg_ad.sh tests/test_linalg_ad_results_formatter.sh tests/test_clean_extern_deps.sh tests/test_setup_extern_tenferro_checkout.sh; do
  env -u OPENBLAS_ROOT -u MKLROOT -u TENFERRO_CPU_FEATURES -u PUBLICATION_GATE_FEATURES bash "$test"
done
bash -n scripts/run_cpu_session.sh scripts/run_route_diagnostic.sh scripts/build_for_tenferro_rev.sh scripts/thread_env.sh scripts/run_cpu_public_api.sh scripts/benchmark_host_idle.sh .devcontainer/install_openblas.sh
