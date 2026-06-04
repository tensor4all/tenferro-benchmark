#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

assert_thread_env_metadata() {
  local openblas_root="$TMP/openblas"
  local output="$TMP/run.yaml"

  mkdir -p "$openblas_root/lib"
  touch "$openblas_root/lib/libopenblas.dylib"

  OMP_NUM_THREADS=1 \
  OMP_THREAD_LIMIT=1 \
  OMP_DYNAMIC=FALSE \
  RAYON_NUM_THREADS=1 \
  TENFERRO_CPU_BACKEND_KIND=blas \
  OPENBLAS_ROOT="$openblas_root" \
  OPENBLAS_NUM_THREADS=1 \
  GOTO_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  VECLIB_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  BLIS_NUM_THREADS=1 \
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    uv run python scripts/collect_run_metadata.py \
      --suite-id cpu/einsum \
      --suite-file benchmarks/cpu/einsum.yaml \
      --timestamp "2026-06-04T12:34:56+09:00" \
      --tenferro-dir extern/tenferro-rs \
      --tenferro-commit abcdef1 \
      --features system-openblas \
      --blas openblas \
      --output "$output"

  uv run python - "$output" <<'PY'
import sys
from pathlib import Path

import yaml

run = yaml.safe_load(Path(sys.argv[1]).read_text())
env = run["environment"]["env"]
expected = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_DYNAMIC": "FALSE",
    "RAYON_NUM_THREADS": "1",
    "TENFERRO_CPU_BACKEND_KIND": "blas",
    "OPENBLAS_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "VECLIB_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}
missing = {key: value for key, value in expected.items() if env.get(key) != value}
if missing:
    raise SystemExit(f"missing or mismatched thread env: {missing}; env={env}")
PY
}

assert_macos_cpu_info() {
  uv run python - <<'PY'
import scripts.collect_cpu_info as cpu

values = {
    "machdep.cpu.brand_string": "Apple M5 Max",
    "hw.physicalcpu": "18",
    "hw.logicalcpu": "18",
    "hw.nperflevels": "2",
    "hw.perflevel0.name": "Super",
    "hw.perflevel0.physicalcpu": "6",
    "hw.perflevel0.logicalcpu": "6",
    "hw.perflevel0.l1icachesize": "196608",
    "hw.perflevel0.l1dcachesize": "131072",
    "hw.perflevel0.l2cachesize": "16777216",
    "hw.perflevel0.cpusperl2": "6",
    "hw.perflevel1.name": "Performance",
    "hw.perflevel1.physicalcpu": "12",
    "hw.perflevel1.logicalcpu": "12",
    "hw.perflevel1.l1icachesize": "131072",
    "hw.perflevel1.l1dcachesize": "65536",
    "hw.perflevel1.l2cachesize": "8388608",
    "hw.perflevel1.cpusperl2": "6",
}

def fake_run(args):
    if args[:2] == ["sysctl", "-n"]:
        return values.get(args[2])
    return None

cpu.platform.system = lambda: "Darwin"
cpu.platform.platform = lambda: "macOS-test-arm64"
cpu._run = fake_run

info = cpu.collect_cpu_info()
assert info["Model"] == "Apple M5 Max", info
assert info["Vendor"] == "Apple", info
assert info["Logical CPUs"] == "18", info
assert info["Physical CPUs"] == "18", info
assert info["Threads per core"] == "1", info
assert "Super: 6 physical / 6 logical" in info["Performance levels"], info
assert "Performance: 12 physical / 12 logical" in info["Performance levels"], info
assert "L2 16 MiB" in info["Performance levels"], info
assert "L2 8 MiB" in info["Performance levels"], info
PY
}

assert_thread_env_metadata
assert_macos_cpu_info
