set -euo pipefail
export CARGO_TARGET_DIR="$PWD/target/compatibility-mkl" CARGO_BUILD_JOBS=8
export LD_LIBRARY_PATH="$MKLROOT/lib:/opt/intel/oneapi/compiler/2026.1/lib:${LD_LIBRARY_PATH:-}"
cargo check --no-default-features --features system-mkl --example _issue_2027_compile_check --example _issue_2028_compile_check --example _issue_2029_compile_check
.venv/bin/python scripts/generate_perf_issue_cases.py --check
.venv/bin/python scripts/validate_benchmark_suite.py benchmarks/cpu/perf_issues.yaml
.venv/bin/python -m unittest tests/test_perf_issues_suite.py tests/test_timing_boundaries.py tests/test_cpu_scope_fixtures.py
.venv/bin/python -m py_compile scripts/cpu_gap_mwe_python.py scripts/confirm_cpu_gaps.py scripts/benchmark_perf_issues.py
cargo test --no-default-features --features system-mkl --bin perf_issue_case
.venv/bin/python - <<'PY'
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,'scripts')
from benchmark_perf_issues import torch_case
cases=[r for r in json.loads(Path('data/instances/perf_issues.json').read_text()) if r['kind']=='cpu_gap_mwe']
assert len(cases)==6
for c in cases:
 if c['backend']=='pytorch-cpu':
  op,check,_,_=torch_case(c);check()
 else:
  args=['target/compatibility-mkl/release/perf_issue_case','--case',c['id'],'--params',json.dumps(c['params']),'--threads','1','--mode','correctness-only','--warmups','0']
  r=json.loads(subprocess.check_output(args,text=True));assert r['correctness_status']=='passed' and not r['samples']
 print(c['id'],'validated')
PY
