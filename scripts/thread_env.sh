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
    # MKL dynamic sizing can cap a declared 4T run to a 2-core hosted runner.
    export MKL_DYNAMIC=FALSE
    export VECLIB_MAXIMUM_THREADS="$threads"
    export VECLIB_NUM_THREADS="$threads"
    export NUMEXPR_NUM_THREADS="$threads"
    export BLIS_NUM_THREADS="$threads"
    export PJRT_NPROC="$threads"
    # JAX 0.10's experimental YNNPACK lowering rejects rank > 8, including
    # intermediates in the standard einsum corpus. Use the regular XLA path.
    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=${xla_multi_thread} --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=${threads}"
    export JULIA_NUM_THREADS="$threads"
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
        MKL_DYNAMIC \
        VECLIB_MAXIMUM_THREADS \
        VECLIB_NUM_THREADS \
        NUMEXPR_NUM_THREADS \
        BLIS_NUM_THREADS \
        PJRT_NPROC \
        XLA_FLAGS \
        JULIA_NUM_THREADS; do
        printf '  %s=%s\n' "$key" "${!key:-}"
    done
}
