#!/usr/bin/env bash
# Run from the benchmark worktree in the recorded OpenBLAS image.
set -euo pipefail
baseline=$(realpath "$1")
final=$(realpath "$2")
out=$(realpath -m "$3")
mkdir -p "$out"
source scripts/thread_env.sh
configure_cpu_thread_env 1
export TENFERRO_MODE=eager TENFERRO_CPU_BACKEND_KIND=blas

profile() {
    local repeat=$1 case=$2 variant binary runs file
    for variant in baseline final; do
        binary=${!variant}
        for runs in 1 3; do
            file="$out/r$repeat-$variant-$case-n$runs"
            BENCH_INSTANCE="$case" BENCH_RUNS="$runs" BENCH_WARMUPS=1 \
                timeout 150s taskset -c 16 valgrind --tool=callgrind \
                --callgrind-out-file="$file.callgrind" "$binary" > "$file.log" 2>&1
            callgrind_annotate --auto=no --inclusive=no --threshold=100 \
                "$file.callgrind" > "$file-annotated.txt"
        done
    done
}
for repeat in 1 2 3; do
    for case in bin_batched_matmul_b32_m128_n128_k128 bin_matmul_1024; do
        profile "$repeat" "$case"
    done
done
profile 1 lm_batch_likelihood_sentence_3_12d
