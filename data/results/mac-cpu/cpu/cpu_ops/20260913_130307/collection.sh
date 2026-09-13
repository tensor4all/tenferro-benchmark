#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source scripts/thread_env.sh
source scripts/cpu_blas_provider.sh
source scripts/benchmark_host_idle.sh
export PUBLICATION_GATE_PROFILE=full PUBLICATION_GATE_TENFERRO_MODE=eager
export TENFERRO_CPU_BACKEND_KIND=blas TENFERRO_CPU_FEATURES=system-accelerate
export BENCHMARK_TARGET_PROFILE=mac-cpu
ensure_blas_env_for_features system-accelerate
run=data/results/mac-cpu/cpu/cpu_ops/$(date +%Y%m%d_%H%M%S)
mkdir -p "$run"
echo "$run" > /tmp/m5-targeted-run-path
cp "${BASH_SOURCE[0]}" "$run/collection.sh"
for threads in 1 4; do
 configure_cpu_thread_env "$threads"
 assert_benchmark_host_idle
 uv run python scripts/collect_run_metadata.py --suite-id cpu/cpu_ops --target-profile mac-cpu --suite-file benchmarks/cpu/einsum.yaml --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --tenferro-dir "$PWD/extern/tenferro-rs" --features system-accelerate --blas accelerate --output "$run/run_t${threads}.yaml"
 target/release/publication_gate > "$run/eager_t${threads}.csv"
 uv run python scripts/collect_run_metadata.py --suite-id cpu/public_api --target-profile mac-cpu --suite-file benchmarks/cpu/public_api.yaml --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --tenferro-dir "$PWD/extern/tenferro-rs" --features system-accelerate --blas accelerate --output "$run/run_views_t${threads}.yaml"
 PUBLIC_API_SUITE_FILTER=cpu/view_metadata PUBLIC_API_EXECUTION_FILTER=direct target/release/benchmark_cpu_public_api --num-threads "$threads" --output "$run/views_t${threads}.csv"
done
