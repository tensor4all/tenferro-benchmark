#!/usr/bin/env bash
# Run in the authorized OpenBLAS container, at the benchmark root.
set -euo pipefail
: "${BASELINE:?2fd8163 binary required}"
: "${CANDIDATE:?636cd499 binary required}"
: "${OUT:?Output directory required}"
mkdir -p "$OUT"
source scripts/thread_env.sh
configure_cpu_thread_env 1
export TENFERRO_MODE=eager BENCH_WARMUPS=1 TENFERRO_OPT_DOT_DECOMPOSER=0
sha256sum "$BASELINE" "$CANDIDATE" > "$OUT/binaries.sha256"
# REPEATS permits bounded sequential collection, without overwriting prior runs.
for repeat in ${REPEATS:-1 2 3}; do
  for variant in baseline candidate; do
    if [[ $variant == baseline ]]; then binary=$BASELINE; else binary=$CANDIDATE; fi
    for case in bin_elementwise_mul_2048x2048 bin_matmul_1024 lm_batch_likelihood_sentence_3_12d; do
      for runs in 1 3; do
        prefix="$OUT/r$repeat-$variant-$case-n$runs"
        [[ ! -e "$prefix.log" ]]
        BENCH_INSTANCE=$case BENCH_RUNS=$runs timeout 150s taskset -c 16 \
          valgrind --tool=callgrind --callgrind-out-file="$prefix.callgrind" \
          "$binary" > "$prefix.log" 2>&1
        callgrind_annotate --auto=no --inclusive=no --threshold=100 \
          "$prefix.callgrind" > "$prefix-annotated.txt"
        printf '%s completed %s\n' "$(date --iso-8601=seconds)" "$prefix"
      done
    done
  done
done
