#!/usr/bin/env bash
# Sequential paired cpu/session_matrix timing on one host with one harness
# (tenferro-rs #1946 B5/B6). Baseline and candidate processes alternate in
# balanced ABBA order; every round keeps its own raw-run directory.
#
# Usage:
#   scripts/run_paired_timing.sh aa <tenferro-rs-dir>
#       A/A noise characterization: the same build in both arms (a, b).
#       Needs BENCH_AA_ROUNDS (even), BENCH_AA_WARMUPS, BENCH_AA_RUNS and
#       BENCH_AA_THREADS (e.g. "1 4"); BENCH_COVERAGE (default quick).
#   scripts/run_paired_timing.sh paired <baseline-tenferro-rs-dir> <candidate-tenferro-rs-dir>
#       Confirmation run. Everything (commits, harness, threads, coverage,
#       repetitions, rounds, thresholds) comes from the declared
#       BENCH_CONFIRM_CONFIG (default benchmarks/cpu/confirmation.yaml); a
#       placeholder config is refused before anything is built. Optional
#       BENCH_AA_DIR names the A/A directory the detector uses.
#
# Environment: BENCHMARK_TARGET_PROFILE (amd-cpu | mac-cpu), TENFERRO_CPU_FEATURES
# (default native). Writes data/results/<profile>/cpu/session_matrix_{aa,paired}/<ts>/
# with plan.json, one run directory per round and arm, and detector output;
# the latest report goes to result/<profile>/cpu/session_matrix_{aa,paired}.md.
# The idle-host guard of run_cpu_session.sh stays enabled; never bypass it for
# these measurements.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
MODE="${1:?mode aa|paired required}"
shift
export BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-amd-cpu}"
export TENFERRO_CPU_FEATURES="${TENFERRO_CPU_FEATURES:-native}"
PYTHON="$PROJECT_DIR/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python3
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
COMMAND="BENCHMARK_TARGET_PROFILE=$BENCHMARK_TARGET_PROFILE TENFERRO_CPU_FEATURES=$TENFERRO_CPU_FEATURES $(printf '%q ' "$0" "$MODE" "$@")"

HARNESS="$(git -C "$PROJECT_DIR" rev-parse HEAD)"

build() { # <tenferro-dir> <var-file>
    "$SCRIPT_DIR/build_for_tenferro_rev.sh" "$1" "$2" benchmark_cpu_session
}

case "$MODE" in
    aa)
        TENFERRO_A="${1:?tenferro-rs dir required}"
        : "${BENCH_AA_ROUNDS:?}" "${BENCH_AA_WARMUPS:?}" "${BENCH_AA_RUNS:?}" "${BENCH_AA_THREADS:?}"
        ROUNDS="$BENCH_AA_ROUNDS"
        read -r -a THREADS <<< "$BENCH_AA_THREADS"
        export BENCH_EFFORT=aa BENCH_COVERAGE="${BENCH_COVERAGE:-quick}"
        ARM_A=a ARM_B=b
        DIR_A="$TENFERRO_A" DIR_B="$TENFERRO_A"
        SUITE=session_matrix_aa
        COMMAND="BENCH_AA_ROUNDS=$ROUNDS BENCH_AA_WARMUPS=$BENCH_AA_WARMUPS BENCH_AA_RUNS=$BENCH_AA_RUNS BENCH_AA_THREADS='$BENCH_AA_THREADS' BENCH_COVERAGE=$BENCH_COVERAGE $COMMAND"
        ;;
    paired)
        DIR_A="${1:?baseline tenferro-rs dir required}"
        DIR_B="${2:?candidate tenferro-rs dir required}"
        CONFIG="${BENCH_CONFIRM_CONFIG:-$PROJECT_DIR/benchmarks/cpu/confirmation.yaml}"
        export BENCH_CONFIRM_CONFIG="$CONFIG" BENCH_EFFORT=confirm
        # Capture first: a failing command substitution inside eval would not stop the script.
        declared="$("$PYTHON" - "$CONFIG" "$DIR_A" "$DIR_B" "$HARNESS" <<'PY'
import subprocess, sys
sys.path.insert(0, "scripts")
import bench_selection
from pathlib import Path
config = bench_selection.load_confirmation_config(Path(sys.argv[1]))
head = lambda d: subprocess.check_output(["git", "-C", d, "rev-parse", "HEAD"], text=True).strip()
checks = [("library.baseline_commit", config["library"]["baseline_commit"], head(sys.argv[2])),
          ("library.candidate_commit", config["library"]["candidate_commit"], head(sys.argv[3])),
          ("harness_commit", config["harness_commit"], sys.argv[4])]
for name, declared, actual in checks:
    if not actual.startswith(str(declared)):
        sys.exit(f"ERROR: {name} is declared {declared} but the checkout is {actual}")
rounds = int(config["repetitions"]["rounds"])
if rounds % 2:
    sys.exit("ERROR: repetitions.rounds must be even for a balanced ABBA order")
print(f"ROUNDS={rounds}")
print("THREADS=(" + " ".join(str(t) for t in config["threads"]) + ")")
print(f"export BENCH_COVERAGE={config['cases']['coverage']}")
if config["cases"].get("case_ids"):
    print("export BENCH_INSTANCE=" + ",".join(config["cases"]["case_ids"]))
if config["build"]["features"] != __import__("os").environ["TENFERRO_CPU_FEATURES"]:
    sys.exit("ERROR: build.features does not match TENFERRO_CPU_FEATURES")
PY
)" || exit 1
        eval "$declared"
        ARM_A=baseline ARM_B=candidate
        SUITE=session_matrix_paired
        COMMAND="BENCH_CONFIRM_CONFIG=$CONFIG ${BENCH_AA_DIR:+BENCH_AA_DIR=$BENCH_AA_DIR }$COMMAND"
        ;;
    *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac

# Generated reports under result/ are outputs, not harness: an A/A run
# rewrites result/<profile>/cpu/session_matrix_aa.md before the paired run.
if [[ -n "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=no -- . ':(exclude)result')" ]]; then
    echo "ERROR: the harness checkout is dirty; paired timing must use one committed harness." >&2
    exit 1
fi
ROOT_DIR="$PROJECT_DIR/data/results/$BENCHMARK_TARGET_PROFILE/cpu/$SUITE/$TIMESTAMP"
mkdir -p "$ROOT_DIR"
VARS_A="$ROOT_DIR/build_${ARM_A}.env"
VARS_B="$ROOT_DIR/build_${ARM_B}.env"
build "$DIR_A" "$VARS_A"
if [[ "$DIR_B" == "$DIR_A" ]]; then cp "$VARS_A" "$VARS_B"; else build "$DIR_B" "$VARS_B"; fi
printf '%s\n' "$COMMAND" > "$ROOT_DIR/command.txt"
if [[ "$MODE" == paired ]]; then cp "$BENCH_CONFIRM_CONFIG" "$ROOT_DIR/confirmation.yaml"; fi

run_arm() { # <arm> <round> <var-file>
    local arm="$1" round="$2" vars="$3" name
    name="$(printf 'r%02d_%s' "$round" "$arm")"
    # shellcheck disable=SC1090
    ( source "$vars"
      BENCH_SESSION_BINARY="$BIN_DIR/benchmark_cpu_session" BENCH_TENFERRO_DIR="$TENFERRO_DIR" \
      BENCH_RUN_DIR="$ROOT_DIR/$name" BENCH_SKIP_LATEST=1 \
          "$SCRIPT_DIR/run_cpu_session.sh" "${THREADS[@]}" ) || echo "round $round $arm exited $?" >&2
    "$PYTHON" - "$ROOT_DIR/plan.json" "$round" "$arm" "$name" "${THREADS[@]}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
plan = json.loads(path.read_text()) if path.exists() else {"order": []}
plan["threads"] = [int(t) for t in sys.argv[5:]]
plan["order"].append({"round": int(sys.argv[2]), "arm": sys.argv[3], "dir": sys.argv[4]})
path.write_text(json.dumps(plan, indent=2) + "\n")
PY
}

for (( round = 1; round <= ROUNDS; round++ )); do
    if (( round % 2 )); then
        run_arm "$ARM_A" "$round" "$VARS_A"; run_arm "$ARM_B" "$round" "$VARS_B"
    else
        run_arm "$ARM_B" "$round" "$VARS_B"; run_arm "$ARM_A" "$round" "$VARS_A"
    fi
done

LATEST="$PROJECT_DIR/result/$BENCHMARK_TARGET_PROFILE/cpu/$SUITE.md"
status=0
if [[ "$MODE" == aa ]]; then
    "$PYTHON" "$SCRIPT_DIR/detect_regressions.py" aa --aa-dir "$ROOT_DIR" \
        --json "$ROOT_DIR/aa.json" --markdown "$ROOT_DIR/report.md" >/dev/null
else
    aa_args=()
    if [[ -n "${BENCH_AA_DIR:-}" ]]; then aa_args=(--aa-dir "$BENCH_AA_DIR"); fi
    "$PYTHON" "$SCRIPT_DIR/detect_regressions.py" confirm --config "$BENCH_CONFIRM_CONFIG" \
        --paired-dir "$ROOT_DIR" ${aa_args[@]+"${aa_args[@]}"} \
        --json "$ROOT_DIR/detector.json" --markdown "$ROOT_DIR/report.md" >/dev/null || status=$?
fi
mkdir -p "$(dirname "$LATEST")"
{ printf '%s\n\n' "- Raw runs: \`${ROOT_DIR#"$PROJECT_DIR"/}\`"
  printf '%s\n\n' "- Command: \`$(cat "$ROOT_DIR/command.txt")\`"
  cat "$ROOT_DIR/report.md"; } > "$LATEST.tmp"
mv "$LATEST.tmp" "$LATEST"
echo "$ROOT_DIR"
exit "$status"
