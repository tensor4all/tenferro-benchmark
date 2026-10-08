#!/usr/bin/env bash
# The paired timing runner must refuse undeclared confirmation configs and
# incomplete A/A requests before building or timing anything.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
link_before="$(readlink extern/tenferro-rs || true)"
runs_before="$(find data/results -maxdepth 4 -type d 2>/dev/null | sort | tr '\n' ' ')"
bash -n scripts/run_paired_timing.sh \
    scripts/build_for_tenferro_rev.sh scripts/run_cpu_session.sh scripts/run_small_work.sh
out="$(BENCH_CONFIRM_CONFIG=benchmarks/cpu/confirmation.yaml \
    scripts/run_paired_timing.sh paired . . 2>&1)" && { echo "placeholder config accepted"; exit 1; }
grep -q "confirmation config is not declared" <<<"$out" || { echo "$out"; exit 1; }
out="$(env -u BENCH_AA_ROUNDS scripts/run_paired_timing.sh aa . 2>&1)" && { echo "aa without repetitions accepted"; exit 1; }
grep -q "BENCH_AA_ROUNDS" <<<"$out" || { echo "$out"; exit 1; }
out="$(scripts/build_for_tenferro_rev.sh . /dev/null 2>&1)" && { echo "non-tenferro dir accepted"; exit 1; }
grep -q "not a tenferro-rs checkout" <<<"$out" || { echo "$out"; exit 1; }
# Nothing was re-pointed, built or collected.
[[ "$(readlink extern/tenferro-rs || true)" == "$link_before" ]] || { echo "extern link changed"; exit 1; }
[[ "$(find data/results -maxdepth 4 -type d 2>/dev/null | sort | tr '\n' ' ')" == "$runs_before" ]] || { echo "a run directory appeared"; exit 1; }
echo "paired timing guards OK"
