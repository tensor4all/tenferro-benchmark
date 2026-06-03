#!/usr/bin/env bash

configure_cpu_thread_env() {
    local threads="${1:?threads required}"
    local xla_multi_thread="false"
    if [[ "$threads" =~ ^[0-9]+$ ]] && (( threads > 1 )); then
        xla_multi_thread="true"
    fi

    export OMP_NUM_THREADS="$threads"
    export OMP_THREAD_LIMIT="$threads"
    export OMP_DYNAMIC=FALSE
    export RAYON_NUM_THREADS="$threads"
    export OPENBLAS_NUM_THREADS="$threads"
    export GOTO_NUM_THREADS="$threads"
    export MKL_NUM_THREADS="$threads"
    export VECLIB_MAXIMUM_THREADS="$threads"
    export NUMEXPR_NUM_THREADS="$threads"
    export BLIS_NUM_THREADS="$threads"
    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=${xla_multi_thread} intra_op_parallelism_threads=${threads}"
}

print_cpu_thread_env() {
    for key in \
        OMP_NUM_THREADS \
        OMP_THREAD_LIMIT \
        OMP_DYNAMIC \
        RAYON_NUM_THREADS \
        OPENBLAS_NUM_THREADS \
        GOTO_NUM_THREADS \
        MKL_NUM_THREADS \
        VECLIB_MAXIMUM_THREADS \
        NUMEXPR_NUM_THREADS \
        BLIS_NUM_THREADS \
        XLA_FLAGS; do
        printf '  %s=%s\n' "$key" "${!key:-}"
    done
}
