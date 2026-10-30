#!/usr/bin/env bash
# Exercise the CPU publication hook without running benchmark timing.
#
# scripts/publish_run.sh renders each run directory's own report and summary and
# then publishes the page from a rendering of the selected runs. The runner only
# needs the uv -> python3 fallback and must propagate a genuine failure.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin" "$TMP/scripts" "$TMP/run" "$TMP/result"
cp "$ROOT/scripts/publish_run.sh" "$TMP/scripts/publish_run.sh"
# uv failure still falls back to python3.
cat > "$TMP/bin/uv" <<'SHIM'
#!/usr/bin/env bash
exit 1
SHIM
chmod +x "$TMP/bin/uv"
cat > "$TMP/scripts/render_run.py" <<'PY'
#!/usr/bin/env python3
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
(run_dir / "report.md").write_text("# CPU Public API Benchmark Results\n")
(run_dir / "public_api.summary.json").write_text('{"schema_version": 2}\n')
PY
cat > "$TMP/scripts/regenerate_report_summaries.py" <<'PY'
#!/usr/bin/env python3
import sys
from pathlib import Path
runs = sys.argv[sys.argv.index("--publish") + 1:]
for run in runs:
    run = Path(run)
    page = Path("result") / "mac-cpu" / "cpu" / "public_api.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text((run / "report.md").read_text())
    (page.parent / "public_api.summary.json").write_text((run / "public_api.summary.json").read_text())
PY
chmod +x "$TMP/scripts/render_run.py" "$TMP/scripts/regenerate_report_summaries.py"
(
  cd "$TMP"
  export PATH="$TMP/bin:$PATH"
  source "$TMP/scripts/publish_run.sh"
  publish_cpu_runs "$TMP" "$TMP/run"
)
test -s "$TMP/run/report.md"
test -s "$TMP/run/public_api.summary.json"
test -s "$TMP/result/mac-cpu/cpu/public_api.md"

# BENCHMARK_RAW_ONLY=1 stops after the run record: no report, summary or page.
rm -rf "$TMP/result" "$TMP/run"
mkdir -p "$TMP/run"
(
  cd "$TMP"
  export PATH="$TMP/bin:$PATH"
  export BENCHMARK_RAW_ONLY=1
  source "$TMP/scripts/publish_run.sh"
  publish_cpu_runs "$TMP" "$TMP/run"
)
test ! -e "$TMP/run/report.md"
test ! -e "$TMP/result"

# A genuine renderer failure must stop the dispatch with a nonzero status.
printf 'raise SystemExit(7)\n' > "$TMP/scripts/render_run.py"
if bash -e -c 'cd "$1"; export PATH="$1/bin:$PATH"; source "$1/scripts/publish_run.sh"; publish_cpu_runs "$1" "$1/run"' \
    fixture "$TMP"; then
  echo 'renderer failure was ignored' >&2
  exit 1
fi
echo "public API publication dispatch test passed"
