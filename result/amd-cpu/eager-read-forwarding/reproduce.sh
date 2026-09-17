#!/usr/bin/env bash
# Run inside the authorized OpenBLAS Docker environment, from the benchmark root.
# Build both revisions using the protocol's identical Cargo flags/local patches.
set -euo pipefail
: "${BASELINE:?Path to the 57bee76 binary required}"
: "${CANDIDATE:?Path to the 2fd8163 binary required}"
: "${OUT:?Fresh output directory required}"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
source scripts/thread_env.sh
configure_cpu_thread_env 1
export TENFERRO_MODE=eager BENCH_WARMUPS=1
sha256sum "$BASELINE" "$CANDIDATE" > "$OUT/binaries.sha256"
for repeat in 1 2 3; do
  for variant in baseline candidate; do
    if [[ $variant == baseline ]]; then binary=$BASELINE; else binary=$CANDIDATE; fi
    for case in bin_elementwise_mul_2048x2048 bin_matmul_1024 lm_batch_likelihood_sentence_3_12d; do
      for runs in 1 3; do
        prefix="$OUT/r$repeat-$variant-$case-n$runs"
        BENCH_INSTANCE=$case BENCH_RUNS=$runs timeout 150s taskset -c 16 \
          valgrind --tool=callgrind --callgrind-out-file="$prefix.callgrind" \
          "$binary" > "$prefix.log" 2>&1
        callgrind_annotate --auto=no --inclusive=no --threshold=100 \
          "$prefix.callgrind" > "$prefix-annotated.txt"
      done
    done
  done
done
