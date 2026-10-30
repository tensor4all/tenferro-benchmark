#!/usr/bin/env bash

# shellcheck source=scripts/cpu_pinning.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cpu_pinning.sh"

configure_cpu_thread_env() {
    local threads="${1:?threads required}"
    # Pinned profiles (zen5-cpu) run each thread row under its taskset mask.
    apply_cpu_affinity "$threads" || return 1
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
        JULIA_NUM_THREADS \
        BENCHMARK_CPU_PINNING \
        BENCHMARK_MEASURED_CPUS \
        BENCHMARK_OBSERVER_CPUS; do
        printf '  %s=%s\n' "$key" "${!key:-}"
    done
}

# The thread-environment lines a report header prints, recorded into the run
# directory at collection time so rendering does not need the collecting shell.
THREAD_ENV_KEYS=(
    OMP_NUM_THREADS
    OMP_THREAD_LIMIT
    OMP_DYNAMIC
    RAYON_NUM_THREADS
    OPENBLAS_NUM_THREADS
    GOTO_NUM_THREADS
    MKL_NUM_THREADS
    VECLIB_MAXIMUM_THREADS
    VECLIB_NUM_THREADS
    NUMEXPR_NUM_THREADS
    BLIS_NUM_THREADS
    XLA_FLAGS
)
# Variables a suite's own report also prints (for example JULIA_NUM_THREADS).
THREAD_ENV_EXTRA_KEYS=()

thread_env_lines() {
    local key
    for key in "${THREAD_ENV_KEYS[@]}" "${THREAD_ENV_EXTRA_KEYS[@]}"; do
        echo "- ${key}: \`${!key:-}\`"
    done
}

# A run directory that holds one run_t<N>.yaml per thread count: start the
# snapshot, then append each row as collection configures it.
begin_thread_envs_snapshot() {
    echo "## Thread Environments" > "${1:?output path required}"
}

append_thread_env_section() {
    local output="${1:?output path required}"
    local threads="${2:?threads required}"
    {
        echo ""
        echo "### Threads: $threads"
        echo ""
        thread_env_lines
    } >> "$output"
}
