#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER="$ROOT_DIR/scripts/run_cpu_public_api.sh"

for variable in PUBLIC_API_EXECUTION_FILTER PUBLIC_API_ATTRIBUTION_OUTPUT; do
    if output="$(
        env \
            "$variable=diagnostic-value" \
            TENFERRO_CPU_BACKEND_KIND=invalid \
            bash "$RUNNER" 1 2>&1
    )"; then
        echo "expected $variable to be rejected by the publication runner" >&2
        exit 1
    fi
    if [[ "$output" != *"$variable is diagnostic-only"* ]]; then
        echo "missing diagnostic-only error for $variable: $output" >&2
        exit 1
    fi
done
