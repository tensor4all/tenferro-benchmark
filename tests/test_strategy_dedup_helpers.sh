#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

(
  cd "$ROOT"
  cargo test --locked --bin tenferro-einsum-benchmark \
    strategy_cache_key_uses_instance_name_and_exact_path
)

uv run python - "$ROOT" <<'PY'
import importlib.util
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location(
    "benchmark_python", root / "scripts" / "benchmark_python.py"
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

instance = {"name": "bin_matmul_1024"}
same = {"path": [[0, 1]], "log2_size": 99.0, "log10_flops": 88.0}
also_same = {"path": [[0, 1]], "log2_size": 1.0, "log10_flops": 2.0}
different = {"path": [[1, 0]], "log2_size": 99.0, "log10_flops": 88.0}

assert module.strategy_cache_key(instance, same) == module.strategy_cache_key(instance, also_same)
assert module.strategy_cache_key(instance, same) != module.strategy_cache_key(instance, different)
assert module.strategy_cache_key(instance, same) != module.strategy_cache_key(
    {"name": "other"}, same
)
PY
