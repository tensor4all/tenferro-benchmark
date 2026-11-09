#!/usr/bin/env bash
# #422: a delegated collector that loses one arm still publishes the rows the
# other arms measured, and reports the failure by its exit status, no GPU needed.
#
# The two arm loops are sourced out of the drivers rather than executed: the
# point under test is that a non-zero arm no longer ends the loop before the
# normalization and formatting steps that turn the surviving rows into a report.
set -euo pipefail
cd "$(dirname "$0")/.."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

# --- stubs shared by both loops -------------------------------------------
run_dir="$TMP/run"
mkdir -p "$run_dir"
rust_build_seconds=0
GPU_ARM_SKIPPED=75
GPU_ARM_PREFIX=()
wall_now() { echo 0; }
wall_phase() { :; }
wall_arm() { :; }
gpu_arm_prepare() { return 0; }
gpu_arm_finish() { :; }
gpu_arm_fail_unless_runnable() { fail "prepare failed unexpectedly: $*"; }

# --- 1. the public-API arm loop -------------------------------------------
sed -n '/^run_arm() {$/,/^}$/p' scripts/run_gpu_public_api.sh > "$TMP/run_arm.sh"
[[ -s "$TMP/run_arm.sh" ]] || fail "run_arm was not found in run_gpu_public_api.sh"
# shellcheck disable=SC1090
source "$TMP/run_arm.sh"

ARM_FAILED=0
outputs=()
arm() {
    local backend="$1" behaviour="$2" output="$run_dir/$1.jsonl"
    shift 2
    case "$behaviour" in
        write) printf '{"backend":"%s"}\n' "$backend" > "$output" ;;
        die) printf '{"backend":"%s"}\n' "$backend" > "$output"; return 3 ;;
        nofile) return 3 ;;
    esac
    return 0
}
run_arm tenferro-cuda-trace "$run_dir/tenferro-cuda-trace.jsonl" arm tenferro-cuda-trace write
run_arm pytorch-cuda "$run_dir/pytorch-cuda.jsonl" arm pytorch-cuda die
run_arm jax-cuda "$run_dir/jax-cuda.jsonl" arm jax-cuda write

[[ "$ARM_FAILED" == 1 ]] || fail "a dead arm must set ARM_FAILED"
[[ ${#outputs[@]} == 3 ]] || fail "every arm that wrote rows is normalized, got ${#outputs[@]}"
[[ -s "$run_dir/tenferro-cuda-trace.jsonl" && -s "$run_dir/pytorch-cuda.jsonl" && -s "$run_dir/jax-cuda.jsonl" ]] \
    || fail "a dead arm's rows were deleted"
# The dead arm comes before the third one: the loop reached it.
grep -q jax-cuda "$run_dir/jax-cuda.jsonl" || fail "the arm after the failure never ran"

# An arm that wrote nothing is not handed to the normalizer.
ARM_FAILED=0
outputs=()
run_arm pytorch-cuda "$run_dir/empty.jsonl" arm pytorch-cuda nofile
[[ "$ARM_FAILED" == 1 ]] || fail "an arm that wrote nothing still marks the suite incomplete"
[[ ${#outputs[@]} == 0 ]] || fail "an empty output must not be normalized"

# --- 2. the linalg AD Python arm loop --------------------------------------
sed -n '/^run_python_arm() {$/,/^}$/p' scripts/run_gpu_linalg_jvp_vjp.sh > "$TMP/run_python_arm.sh"
[[ -s "$TMP/run_python_arm.sh" ]] || fail "run_python_arm was not found in run_gpu_linalg_jvp_vjp.sh"

RESULT_JSONL="$TMP/all.jsonl"
: > "$RESULT_JSONL"
RUN_DIR="$TMP/run"
DEVICE_ORDINAL=0
SUITE_FILE="$TMP/suite.yaml"
: > "$SUITE_FILE"
# The arm body is one process; stub it with a shell function of the same name so
# the extracted loop runs it without a GPU.
PYTHON=()
SCRIPT_DIR="$TMP/fake"
mkdir -p "$SCRIPT_DIR"
cat > "$SCRIPT_DIR/benchmark_gpu_linalg_ad.py" <<'SH'
#!/usr/bin/env bash
# argv: <chunk> <device> "" <backend> -- <suite>
chunk="$1"
printf '{"arm":"%s"}\n' "$4" > "$chunk"
[[ "$4" != "pytorch-cuda" ]] || exit 3
SH
chmod +x "$SCRIPT_DIR/benchmark_gpu_linalg_ad.py"
PYTHON=(bash)
# shellcheck disable=SC1090
source "$TMP/run_python_arm.sh"

ARM_FAILED=0
for backend in tenferro-cuda-eager pytorch-cuda jax-cuda; do
    chunk="$TMP/chunk_$backend.jsonl"
    : > "$chunk"
    run_python_arm "$backend" "$chunk"
done
[[ "$ARM_FAILED" == 1 ]] || fail "a dead AD arm must set ARM_FAILED"
[[ "$(wc -l < "$RESULT_JSONL")" == 3 ]] || fail "every AD arm's rows reach the suite: $(wc -l < "$RESULT_JSONL")"

echo "delegated arm continuation test passed"
