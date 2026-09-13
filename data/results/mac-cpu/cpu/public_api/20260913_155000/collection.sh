#!/bin/bash
set -euo pipefail
cd /Users/hiroshi/projects/tensor4all/tenferro-benchmark-m5-refresh
export PATH="/Users/hiroshi/.juliaup/bin:$PATH"
export BENCHMARK_TARGET_PROFILE=mac-cpu PUBLICATION_GATE_PROFILE=full TENFERRO_CPU_FEATURES=system-accelerate TENFERRO_CPU_BACKEND_KIND=blas
export BENCHMARK_COMMIT="$(git rev-parse HEAD)"
source scripts/thread_env.sh
source scripts/benchmark_host_idle.sh
RUN_DIR="data/results/mac-cpu/cpu/public_api/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
cp /tmp/tenferro-rootcause/final_followup.sh "$RUN_DIR/collection.sh"
for t in 1 4; do
 configure_cpu_thread_env "$t"
 .venv/bin/python scripts/collect_run_metadata.py --suite-id cpu/public_api --target-profile mac-cpu \
  --suite-file benchmarks/cpu/public_api.yaml --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --tenferro-dir extern/tenferro-rs --features system-accelerate --blas accelerate --output "$RUN_DIR/run_t${t}.yaml"
 assert_benchmark_host_idle
 PUBLIC_API_SUITE_FILTER=cpu/view_metadata julia --project=. scripts/benchmark_cpu_public_api_julia.jl \
  --num-threads "$t" --output "$RUN_DIR/julia_views_t${t}.csv"
done
.venv/bin/python - "$RUN_DIR" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,'scripts')
from format_cpu_ops_results import format_table
patch=Path(sys.argv[1]);base=Path('data/results/mac-cpu/cpu/public_api/20260913_154011')
paths=sorted(base.glob('cpu_public_api_t*.csv'))+sorted(patch.glob('julia_views_t*.csv'))
(patch/'source_manifest.json').write_text(json.dumps({'base_run':str(base),'replacement':'Only Julia cpu/view_metadata rows replace the corresponding base rows. Other timing/status values are reused unchanged.','csv_sources_in_override_order':[str(p) for p in paths]},indent=2)+'\n')
latest=Path('result/mac-cpu/cpu/public_api.md')
prefix=latest.read_text().split('## CPU Benchmark Items')[0]
prefix += f'## Julia view batching correction\n\nThe Julia view rows were recollected at 16 calls per interval using benchmark commit 4c63ae9. Correction metadata and raw rows: `{patch}`. All other rows retain the full run above. The source manifest records CSV replacement order; original raw data remain unchanged.\n\n'
report=prefix+format_table(paths)
latest.write_text(report);(patch/'report.md').write_text(report)
print('Julia view correction:',patch)
PY
assert_benchmark_host_idle
./scripts/run_cpu_session.sh 1 4
