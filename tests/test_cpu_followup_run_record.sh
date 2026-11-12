#!/usr/bin/env bash
# The follow-up collector records a run record next to its --output results file,
# in the same metadata shape as every other suite, without changing that file.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/project/scripts" "$TMP/project/mwe/cpu_followup" \
         "$TMP/project/benchmarks/cpu" "$TMP/project/pins" \
         "$TMP/project/.venv/bin" "$TMP/project/target"
for name in scripts/run_cpu_followup.sh scripts/tenferro_checkout.sh \
            scripts/python_venv.sh scripts/thread_env.sh scripts/cpu_pinning.sh \
            scripts/benchmark_host_idle.sh scripts/cpu_blas_provider.sh scripts/link_preflight.sh \
            scripts/collect_run_metadata.py scripts/collect_gpu_info.py \
            scripts/benchmark_layout.py scripts/validate_benchmark_suite.py \
            benchmarks/cpu_pinning_rows.tsv; do
    cp "$ROOT/$name" "$TMP/project/$name"
done
cp -R "$ROOT/schemas" "$TMP/project/schemas"
printf '%s\n' "unused" > "$TMP/project/pins/tenferro-rs.rev"

# A real git checkout so the recorded commit is the one a run built from.
mkdir -p "$TMP/project/extern/tenferro-rs"
git -C "$TMP/project/extern/tenferro-rs" init -q
git -C "$TMP/project/extern/tenferro-rs" config user.email "benchmark-test@example.com"
git -C "$TMP/project/extern/tenferro-rs" config user.name "Benchmark Test"
printf 'x\n' > "$TMP/project/extern/tenferro-rs/marker.txt"
git -C "$TMP/project/extern/tenferro-rs" add marker.txt
git -C "$TMP/project/extern/tenferro-rs" commit -qm init
COMMIT="$(git -C "$TMP/project/extern/tenferro-rs" rev-parse HEAD)"
# The harness venv interpreter, used to render the run record for real.
REAL_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')"

# MKLROOT only has to resolve for the blas-mkl feature the collector declares.
mkdir -p "$TMP/mkl/include" "$TMP/mkl/lib"
printf '#define __INTEL_MKL__ 2026\n#define __INTEL_MKL_MINOR__ 0\n#define __INTEL_MKL_UPDATE__ 0\n' \
    > "$TMP/mkl/include/mkl_version.h"

# The stub stands in for the MWE driver and records what it was asked to run.
cat > "$TMP/project/.venv/bin/python" <<'PY'
#!/usr/bin/env bash
# The real collector writes the run record (it needs the harness venv's yaml);
# only the MWE driver runs is stubbed out.
for argument in "$@"; do
    if [[ "$argument" == */collect_run_metadata.py ]]; then
        exec "$CPU_FOLLOWUP_REAL_PYTHON" "$@"
    fi
done
printf '%s\n' "$*" > "$CPU_FOLLOWUP_STUB_LOG"
PY
chmod +x "$TMP/project/.venv/bin/python"

(
    cd "$TMP/project"
    export MKLROOT="$TMP/mkl" OPENBLAS_ROOT="$TMP/mkl"
    export BENCHMARK_TARGET_PROFILE=amd-cpu
    export CPU_FOLLOWUP_STUB_LOG="$TMP/driver.log"
    export CPU_FOLLOWUP_REAL_PYTHON="$REAL_PYTHON"
    # A metadata-shape test with a stubbed driver, never a measurement.
    export BENCHMARK_ALLOW_BUSY_HOST=1
    # No MWE binary is built or run; the driver is the stub above.
    bash scripts/run_cpu_followup.sh 1 --output "$TMP/followup.jsonl" --case some.case
)

test -f "$TMP/followup.jsonl.run.yaml"
grep -q "cpu_followup_cases.py --threads 1 --output $TMP/followup.jsonl --case some.case" "$TMP/driver.log"
grep -q "environment:" "$TMP/followup.jsonl.run.yaml"
grep -qE "container: (devcontainer|none)" "$TMP/followup.jsonl.run.yaml"
grep -q "commit: $COMMIT" "$TMP/followup.jsonl.run.yaml"
grep -q "suite_id: cpu/followup" "$TMP/followup.jsonl.run.yaml"
uv run python scripts/validate_benchmark_suite.py --kind run "$TMP/followup.jsonl.run.yaml"

# The MWE binary is built by --prepare at the revision current then; a collection
# after the checkout moved must still record what was built.
PREPARED_COMMIT="$COMMIT"
# --prepare builds the MWE here and stamps the revision into the build directory.
mkdir -p "$TMP/bin" "$TMP/project/target/followup-current"
printf '#!/usr/bin/env bash\nexit 0\n' > "$TMP/bin/cargo"
chmod +x "$TMP/bin/cargo"
# The link preflight runs before the MWE build; here it is a stub that records how it was asked
# (the real check is campaign-toolkit's, exercised by the validate CI job).
printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$*" >> "%s"\nexit 0\n' "$TMP/preflight.log" > "$TMP/bin/preflight-python"
chmod +x "$TMP/bin/preflight-python"
export LINK_PREFLIGHT_PYTHON="$TMP/bin/preflight-python"
(
    cd "$TMP/project"
    # No uv on PATH, so the prepare step does not touch a Python environment.
    export PATH="$TMP/bin:/usr/bin:/bin"
    export MKLROOT="$TMP/mkl" GPU_BENCH_NO_PULL=1
    bash scripts/run_cpu_followup.sh --prepare
)
grep -q -- '-m campaign_toolkit.link_preflight .*--arm followup-mwe' "$TMP/preflight.log" || {
    echo "run_cpu_followup.sh --prepare built the MWE without the link preflight" >&2; exit 1; }
STAMP="$TMP/project/target/followup-current/.tenferro-commit"
test -s "$STAMP"
test "$(cat "$STAMP")" = "$PREPARED_COMMIT"

printf 'moved\n' > "$TMP/project/extern/tenferro-rs/marker.txt"
git -C "$TMP/project/extern/tenferro-rs" commit -qam moved
MOVED_COMMIT="$(git -C "$TMP/project/extern/tenferro-rs" rev-parse HEAD)"
test "$MOVED_COMMIT" != "$PREPARED_COMMIT"

# --prepare removed the stubbed interpreter; put it back for the collection.
mkdir -p "$TMP/project/.venv/bin"
cat > "$TMP/project/.venv/bin/python" <<'PY'
#!/usr/bin/env bash
for argument in "$@"; do
    if [[ "$argument" == */collect_run_metadata.py ]]; then
        exec "$CPU_FOLLOWUP_REAL_PYTHON" "$@"
    fi
done
printf '%s\n' "$*" > "$CPU_FOLLOWUP_STUB_LOG"
PY
chmod +x "$TMP/project/.venv/bin/python"

(
    cd "$TMP/project"
    export MKLROOT="$TMP/mkl" OPENBLAS_ROOT="$TMP/mkl"
    export BENCHMARK_TARGET_PROFILE=amd-cpu
    export CPU_FOLLOWUP_STUB_LOG="$TMP/driver2.log"
    export CPU_FOLLOWUP_REAL_PYTHON="$REAL_PYTHON"
    export BENCHMARK_ALLOW_BUSY_HOST=1
    bash scripts/run_cpu_followup.sh 1 --output "$TMP/stamped.jsonl" --case some.case
)
grep -q "commit: $PREPARED_COMMIT" "$TMP/stamped.jsonl.run.yaml"
if grep -q "commit: $MOVED_COMMIT" "$TMP/stamped.jsonl.run.yaml"; then
    echo "the run record reported the moved checkout instead of the built revision" >&2
    exit 1
fi
uv run python scripts/validate_benchmark_suite.py --kind run "$TMP/stamped.jsonl.run.yaml"

echo "cpu followup run record ok"
