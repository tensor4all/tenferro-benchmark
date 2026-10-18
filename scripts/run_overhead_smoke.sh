#!/usr/bin/env bash
# cpu/overhead_smoke (tenferro-benchmark#197): about ten small public
# tenferro-rs calls on one run thread, each reporting ns/op and allocs/op for
# the nightly trend of tenferro-rs #2178. Always single-threaded (rows that
# compare backend thread budgets set them explicitly); a few minutes at most.
# Optional BENCH_INSTANCE=id[,id...] filter. Writes
# data/results/<target_profile>/cpu/overhead_smoke/<timestamp>/ and
# result/<target_profile>/cpu/overhead_smoke.md. No row gates.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
THREADS=1
source "$SCRIPT_DIR/cpu_blas_provider.sh"
source "$SCRIPT_DIR/thread_env.sh"
source "$SCRIPT_DIR/benchmark_host_idle.sh"
export TENFERRO_CPU_FEATURES="$(normalize_cpu_blas_features "${TENFERRO_CPU_FEATURES:-}")"
export BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-amd-cpu}"
if [[ "${SKIP_EXTERN_SETUP:-0}" != 1 ]]; then
    source "$SCRIPT_DIR/setup_extern_deps.sh"
fi
ensure_blas_env_for_features "$TENFERRO_CPU_FEATURES"
PYTHON="$PROJECT_DIR/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python3
"$PYTHON" "$SCRIPT_DIR/generate_overhead_smoke_cases.py" --check
"$PYTHON" "$SCRIPT_DIR/validate_benchmark_suite.py" benchmarks/cpu/overhead_smoke.yaml
# Build once before sequential timing collection.
cargo build --release --no-default-features --features "$TENFERRO_CPU_FEATURES" --bin perf_issue_case
BINARY="${CARGO_TARGET_DIR:-$PROJECT_DIR/target}/release/perf_issue_case"
TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
"$PYTHON" -c 'from scripts.benchmark_layout import safe_target_profile; import sys; safe_target_profile(sys.argv[1])' "$BENCHMARK_TARGET_PROFILE"
RUN_DIR="$PROJECT_DIR/data/results/$BENCHMARK_TARGET_PROFILE/cpu/overhead_smoke/$TIMESTAMP"
mkdir -p "$RUN_DIR"
"$PYTHON" "$SCRIPT_DIR/collect_cpu_info.py" --markdown > "$RUN_DIR/cpu_info.md"
export BENCHMARK_COMMIT="${BENCHMARK_COMMIT:-$(git rev-parse HEAD)}"
export BENCH_COMMAND="${BENCH_COMMAND:-$(for v in BENCHMARK_TARGET_PROFILE TENFERRO_CPU_FEATURES BENCH_INSTANCE; do if [[ -n "${!v:-}" ]]; then printf '%s=%q ' "$v" "${!v}"; fi; done)$(printf '%q ' "$0" "$@")}"
configure_cpu_thread_env "$THREADS"
"$PYTHON" "$SCRIPT_DIR/collect_run_metadata.py" \
    --suite-id cpu/overhead_smoke --target-profile "$BENCHMARK_TARGET_PROFILE" \
    --suite-file benchmarks/cpu/overhead_smoke.yaml --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --tenferro-dir "$PROJECT_DIR/extern/tenferro-rs" --features "$TENFERRO_CPU_FEATURES" \
    --blas "$(blas_impl_for_features "$TENFERRO_CPU_FEATURES")" --output "$RUN_DIR/run_t${THREADS}.yaml"
"$PYTHON" - "$RUN_DIR/run_t${THREADS}.yaml" <<'PY'
import json, os, sys, subprocess, yaml
from scripts.generate_overhead_smoke_cases import collection_policy
from pathlib import Path
path = Path(sys.argv[1])
metadata = yaml.safe_load(path.read_text())
metadata['environment'].setdefault('env', {}).update({
    'BENCHMARK_COMMIT': os.environ['BENCHMARK_COMMIT'], 'BENCHMARK_BUILD_PROFILE': 'release',
    'RUSTC_VERSION': subprocess.check_output(['rustc', '--version'], text=True).strip(),
    'BENCH_INSTANCE': os.environ.get('BENCH_INSTANCE') or None})
cases = json.loads(Path('data/instances/overhead_smoke.json').read_text())
selected = {item.strip() for item in os.environ.get('BENCH_INSTANCE', '').split(',') if item.strip()}
if selected:
    cases = [case for case in cases if case['id'] in selected]
metadata['collection'] = {'command': os.environ.get('BENCH_COMMAND', ''), **collection_policy(cases)}
path.write_text(yaml.safe_dump(metadata, sort_keys=False))
PY
cp "$RUN_DIR/run_t${THREADS}.yaml" "$RUN_DIR/run.yaml"
assert_benchmark_host_idle
failed=0
"$PYTHON" "$SCRIPT_DIR/benchmark_overhead_smoke.py" --binary "$BINARY" \
    --threads "$THREADS" --output "$RUN_DIR/samples_t${THREADS}.jsonl" || failed=1
"$PYTHON" "$SCRIPT_DIR/benchmark_overhead_smoke.py" --report "$RUN_DIR" --target-profile "$BENCHMARK_TARGET_PROFILE"
exit "$failed"
