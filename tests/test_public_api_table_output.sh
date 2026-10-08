#!/usr/bin/env bash
# Exercise the actual formatter dispatch without running benchmark timing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin" "$TMP/scripts"
python3 - "$ROOT/scripts/run_cpu_public_api.sh" "$TMP/dispatch.sh" <<'PY'
from pathlib import Path
import sys
source = Path(sys.argv[1]).read_text()
end = source.index('\n{\n    echo "# CPU Public API Benchmark Results"')
start = source.rfind('if command -v uv', 0, end)
Path(sys.argv[2]).write_text(source[start:end])
PY
# uv failure still falls back to python; a runner stdout failure in tee must
# have no effect because reports are written directly to the artifact file.
cat > "$TMP/bin/uv" <<'SHIM'
#!/usr/bin/env bash
exit 1
SHIM
cat > "$TMP/bin/tee" <<'SHIM'
#!/usr/bin/env bash
echo 'tee: stdout: Resource temporarily unavailable' >&2
exit 1
SHIM
chmod +x "$TMP/bin/uv" "$TMP/bin/tee"
cat > "$TMP/scripts/format_cpu_ops_results.py" <<'PY'
print('row\n' * 50000, end='')
PY
export PATH="$TMP/bin:$PATH"
SCRIPT_DIR="$TMP/scripts"
TABLE="$TMP/table.md"
CSVS=(fixture.csv)
source "$TMP/dispatch.sh" > "$TMP/stdout.log"
[[ "$(wc -c < "$TABLE")" -eq 200000 ]]
[[ ! -s "$TMP/stdout.log" ]]
# Genuine formatter failure must still stop the dispatch with a nonzero status.
printf 'raise SystemExit(7)\n' > "$TMP/scripts/format_cpu_ops_results.py"
if bash -e -c 'SCRIPT_DIR="$1"; TABLE="$2"; CSVS=(fixture.csv); source "$3"' \
    fixture "$SCRIPT_DIR" "$TABLE" "$TMP/dispatch.sh"; then
  echo 'formatter failure was ignored' >&2
  exit 1
fi
