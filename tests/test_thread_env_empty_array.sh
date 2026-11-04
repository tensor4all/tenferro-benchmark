#!/usr/bin/env bash
set -euo pipefail

# Regression: macOS /bin/bash 3.2 treats `"${arr[@]}"` of an empty array as an
# unbound variable under `set -u`, which aborted every CPU collection on the
# mac runner (thread_env.sh: THREAD_ENV_EXTRA_KEYS). Run this with
# /bin/bash on macOS to reproduce the original crash.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck source=scripts/thread_env.sh
source "$ROOT/scripts/thread_env.sh"

# Default: the extra-key array is empty, and the base keys still print.
THREAD_ENV_EXTRA_KEYS=()
default_lines="$(OMP_NUM_THREADS=4 thread_env_lines)"
grep -qxF -- '- OMP_NUM_THREADS: `4`' <<< "$default_lines"
if grep -q -- '- JULIA_NUM_THREADS:' <<< "$default_lines"; then
  echo "an empty THREAD_ENV_EXTRA_KEYS must not contribute a key" >&2
  exit 1
fi

# A suite that declares extra keys still gets them.
THREAD_ENV_EXTRA_KEYS=(JULIA_NUM_THREADS)
extra_lines="$(JULIA_NUM_THREADS=4 thread_env_lines)"
grep -qxF -- '- JULIA_NUM_THREADS: `4`' <<< "$extra_lines"

# An entirely unset extra-key array must stay safe too.
unset THREAD_ENV_EXTRA_KEYS
unset_lines="$(OMP_NUM_THREADS=1 thread_env_lines)"
grep -qxF -- '- OMP_NUM_THREADS: `1`' <<< "$unset_lines"

echo "thread_env empty-array: ok ($(bash --version | head -1))"
