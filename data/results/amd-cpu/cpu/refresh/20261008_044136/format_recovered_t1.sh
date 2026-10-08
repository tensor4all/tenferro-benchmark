#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$PWD"
export PATH="$PWD/.venv/bin:$PATH"
SCRIPT_DIR="$PWD/scripts"
NUM_THREADS=1
HAVE_JULIA=1
CPU_SUITE_ID=cpu/einsum
CPU_SUITE_FILE="$PWD/benchmarks/cpu/einsum.yaml"
BENCHMARK_TIMESTAMP=20261008_044146
CPU_RUN_DIR="$PWD/data/results/amd-cpu/cpu/einsum/$BENCHMARK_TIMESTAMP"
CPU_RUN_YAML="$CPU_RUN_DIR/run.yaml"
TENFERRO_COMMIT="$(git -C extern/tenferro-rs rev-parse HEAD)"
source scripts/thread_env.sh
source scripts/cpu_blas_provider.sh
configure_cpu_thread_env 1
source data/results/amd-cpu/cpu/refresh/20261008_044136/report_functions.sh
write_run_metadata "$CPU_RUN_YAML" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 - "$CPU_RUN_YAML" <<'PYMETA'
import sys,yaml
from pathlib import Path
p=Path(sys.argv[1]);data=yaml.safe_load(p.read_text())
old=yaml.safe_load((p.parent/'einsum_run.yaml').read_text())
data['collection_recovery']={'einsum_original_metadata':'einsum_run.yaml','einsum_harness_commit':old['harness']['commit'],'cpu_ops_harness_commit':data['harness']['commit'],'scope':'Completed einsum retained; failed CPU ops rerun after consumed-descriptor restoration fix.'}
p.write_text(yaml.safe_dump(data,sort_keys=False))
PYMETA
TENFERRO_TRACE_LOG="$CPU_RUN_DIR/tenferro_trace_t1_$BENCHMARK_TIMESTAMP.log"
TENFERRO_EAGER_LOG="$CPU_RUN_DIR/tenferro_eager_t1_$BENCHMARK_TIMESTAMP.log"
PYTORCH_LOG="$CPU_RUN_DIR/pytorch_cpu_t1_$BENCHMARK_TIMESTAMP.log"
JAX_LOG="$CPU_RUN_DIR/jax_cpu_t1_$BENCHMARK_TIMESTAMP.log"
JULIA_LOG="$CPU_RUN_DIR/julia_omeinsum_t1_$BENCHMARK_TIMESTAMP.log"
MARKDOWN_TABLE="$CPU_RUN_DIR/einsum_table_t1_$BENCHMARK_TIMESTAMP.md"
CPU_OPS_LOG="$CPU_RUN_DIR/cpu_ops_t1_$BENCHMARK_TIMESTAMP.csv"
CPU_OPS_MD="$CPU_RUN_DIR/cpu_ops_t1_$BENCHMARK_TIMESTAMP.md"
LINALG_AD_MD="$CPU_RUN_DIR/linalg_jvp_vjp_t1_$BENCHMARK_TIMESTAMP.md"
python3 scripts/format_results.py "$TENFERRO_TRACE_LOG" "$TENFERRO_EAGER_LOG" "$PYTORCH_LOG" "$JAX_LOG" "$JULIA_LOG" > "$MARKDOWN_TABLE"
python3 scripts/format_cpu_ops_results.py "$CPU_OPS_LOG" > "$CPU_OPS_MD"
python3 scripts/format_linalg_ad_results.py "$CPU_OPS_LOG" > "$LINALG_AD_MD"
write_einsum_report "$CPU_RUN_DIR/report.md"
write_cpu_report "$CPU_RUN_DIR/cpu_ops_report.md"
write_linalg_ad_report "$CPU_RUN_DIR/linalg_jvp_vjp_report.md"
for report in report cpu_ops_report linalg_jvp_vjp_report; do
    printf '\n## Collection recovery\n\nCompleted einsum measurements retained from the initial collection; CPU ops were recollected after the input-restoration fix. Component commits and original metadata are recorded in `run.yaml` and `einsum_run.yaml`. Exact commands: `data/results/amd-cpu/cpu/refresh/20261008_044136/commands_remaining.sh`.\n' >> "$CPU_RUN_DIR/$report.md"
done
