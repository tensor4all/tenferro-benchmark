#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/public_api_selection.sh"
PUBLIC_API_CANDIDATES_ONLY=1
select_public_api_group cpu/elementwise_reduction "add,select,where_select"
[[ "$PUBLIC_API_GROUP_BENCHMARK_FILTER" == where_select ]]
if select_public_api_group cpu/elementwise_reduction "add,select"; then
    echo "unselected operations must not execute" >&2
    exit 1
fi
select_public_api_group cpu/linalg_uncovered
[[ "$PUBLIC_API_GROUP_BENCHMARK_FILTER" == householder_qr,rank_revealing_qr,rank_revealing_qr_deficient ]]
select_public_api_group cpu/output_reuse
[[ "$PUBLIC_API_GROUP_BENCHMARK_FILTER" == solve_read_into ]]
if select_public_api_group cpu/complex; then exit 1; fi
PUBLIC_API_CANDIDATES_ONLY=0
select_public_api_group cpu/complex "conj,mul"
[[ "$PUBLIC_API_GROUP_BENCHMARK_FILTER" == conj,mul ]]
PUBLIC_API_FOLLOWUPS_ONLY=1
select_public_api_group cpu/neural_public_api
[[ -z "$PUBLIC_API_GROUP_BENCHMARK_FILTER" ]]
if select_public_api_group cpu/linalg_uncovered; then exit 1; fi
