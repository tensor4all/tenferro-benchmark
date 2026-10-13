#!/usr/bin/env bash

# Shared by Rust and JAX collection groups. Keep focused collection isolated
# from the complete cpu/public_api latest report.
select_public_api_group() {
    local suite="$1" requested="${2:-}" names name
    PUBLIC_API_GROUP_BENCHMARK_FILTER="$requested"
    if [[ "${PUBLIC_API_FOLLOWUPS_ONLY:-0}" == "1" ]]; then
        [[ "$suite" == "cpu/neural_public_api" ]] || return 1
        return 0
    fi
    [[ "${PUBLIC_API_CANDIDATES_ONLY:-0}" == "1" ]] || return 0
    case "$suite" in
        cpu/elementwise_reduction) names="reduce_sum_squares_axis0,masked_log_softmax_axis1,where_select" ;;
        cpu/linalg_uncovered) names="householder_qr,rank_revealing_qr,rank_revealing_qr_deficient" ;;
        cpu/output_reuse) names="solve_read_into" ;;
        *) return 1 ;;
    esac
    PUBLIC_API_GROUP_BENCHMARK_FILTER=""
    local IFS=,
    for name in $names; do
        if [[ -z "$requested" || ",$requested," == *",$name,"* ]]; then
            PUBLIC_API_GROUP_BENCHMARK_FILTER+="${PUBLIC_API_GROUP_BENCHMARK_FILTER:+,}$name"
        fi
    done
    [[ -n "$PUBLIC_API_GROUP_BENCHMARK_FILTER" ]]
}
