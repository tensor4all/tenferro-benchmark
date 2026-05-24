#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# tenferro uses its native trailing-batch layout, while Torch/JAX use their
# native leading-batch layout. The generated benchmark labels must not imply
# that all backends use rightmost batch axes.
grep -q 'native batch layout' "$ROOT/src/bin/publication_gate.rs"
grep -q 'native batch layout' "$ROOT/scripts/benchmark_cpu_ops_python.py"
grep -q 'native batch layout' "$ROOT/cpp/benchmark_cpu_ops_libtorch.cpp"

! grep -q 'rightmost batch' "$ROOT/scripts/benchmark_cpu_ops_python.py"
! grep -q 'rightmost batch' "$ROOT/cpp/benchmark_cpu_ops_libtorch.cpp"
