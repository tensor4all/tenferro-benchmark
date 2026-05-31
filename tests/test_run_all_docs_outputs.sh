#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/scripts" "$TMP/data/results" "$TMP/extern/tenferro-rs"
cp "$ROOT/scripts/run_all.sh" "$TMP/scripts/run_all.sh"
cp "$ROOT/scripts/collect_cpu_info.py" "$TMP/scripts/collect_cpu_info.py"

(
  cd "$TMP/extern/tenferro-rs"
  git init -q
  git config user.email test@example.invalid
  git config user.name "Test User"
  printf 'fixture\n' > README.md
  git add README.md
  git commit -q -m 'fixture commit'
)
TENFERRO_COMMIT="$(git -C "$TMP/extern/tenferro-rs" rev-parse HEAD)"

cat > "$TMP/scripts/setup_extern_deps.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
printf 'called\n' > "$root/setup_extern_called"
export OPENBLAS_ROOT="$root/openblas"
export TENFERRO_RS_DIR="$root/extern/tenferro-rs"
export Torch_DIR="$root/extern/pytorch-openblas/torch/share/cmake/Torch"
SH

cat > "$TMP/scripts/run_all_rust.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
threads="${1:-1}"
root="$(cd "$(dirname "$0")/.." && pwd)"
: "${BENCHMARK_TIMESTAMP:?}"
printf 'tenferro-trace\nBackend: tenferro-trace\nOMP_NUM_THREADS=%s\nRAYON_NUM_THREADS=%s\nStrategy: opt_flops\nInstance Tensors log10FLOPS log2SIZE Median (ms) IQR (ms) Compile (ms)\nbin_matmul_256 2 0 0 1.0 0.1 0.0\n' "$threads" "$threads" > "$root/data/results/tenferro_trace_t${threads}_${BENCHMARK_TIMESTAMP}.log"
printf 'tenferro-eager\nBackend: tenferro-eager\nOMP_NUM_THREADS=%s\nRAYON_NUM_THREADS=%s\nStrategy: opt_flops\nInstance Tensors log10FLOPS log2SIZE Median (ms) IQR (ms) Compile (ms)\nbin_matmul_256 2 0 0 1.2 0.1 0.0\n' "$threads" "$threads" > "$root/data/results/tenferro_eager_t${threads}_${BENCHMARK_TIMESTAMP}.log"
SH

cat > "$TMP/scripts/run_all_libtorch.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
threads="${1:-1}"
root="$(cd "$(dirname "$0")/.." && pwd)"
: "${BENCHMARK_TIMESTAMP:?}"
printf 'libtorch-cpu einsum benchmark suite\nOMP_NUM_THREADS=%s\nStrategy: opt_flops\nInstance Tensors log10FLOPS log2SIZE Median (ms) IQR (ms)\nbin_matmul_256 2 0 0 0.8 0.1\n' "$threads" > "$root/data/results/libtorch_cpu_t${threads}_${BENCHMARK_TIMESTAMP}.log"
SH

cat > "$TMP/scripts/run_all_python.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
threads="${1:-1}"
root="$(cd "$(dirname "$0")/.." && pwd)"
: "${BENCHMARK_TIMESTAMP:?}"
printf 'pytorch-cpu einsum benchmark suite\nOMP_NUM_THREADS=%s\nStrategy: opt_flops\nInstance Tensors log10FLOPS log2SIZE Median (ms) IQR (ms)\nbin_matmul_256 2 0 0 0.7 0.1\n' "$threads" > "$root/data/results/pytorch_cpu_t${threads}_${BENCHMARK_TIMESTAMP}.log"
printf 'jax-cpu einsum benchmark suite\nOMP_NUM_THREADS=%s\nStrategy: opt_flops\nInstance Tensors log10FLOPS log2SIZE Median (ms) IQR (ms)\nbin_matmul_256 2 0 0 0.5 0.1\n' "$threads" > "$root/data/results/jax_cpu_t${threads}_${BENCHMARK_TIMESTAMP}.log"
SH

cat > "$TMP/scripts/run_cpu_ops.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
threads="${1:-1}"
root="$(cd "$(dirname "$0")/.." && pwd)"
: "${BENCHMARK_TIMESTAMP:?}"
printf 'suite,benchmark,dtype,threads,shape,backend,median_ms,iqr_ms,status\nmatmul,f64_square,f64,%s,2x2,tenferro-eager,1.000,0.100,ok\nmatmul,f64_square,f64,%s,2x2,tenferro-trace,1.100,0.100,ok\nmatmul,f64_square,f64,%s,2x2,libtorch-cpu,0.900,0.100,ok\nmatmul,f64_square,f64,%s,2x2,pytorch-cpu,0.800,0.100,ok\nmatmul,f64_square,f64,%s,2x2,jax-cpu,0.700,0.100,ok\n' "$threads" "$threads" "$threads" "$threads" "$threads" > "$root/data/results/cpu_ops_t${threads}_${BENCHMARK_TIMESTAMP}.csv"
SH

cat > "$TMP/scripts/format_results.py" <<'PY'
#!/usr/bin/env python3
import sys
args = "\n".join(sys.argv[1:])
threads = "4" if "_t4_" in args else "1"
assert "tenferro_trace" in args, args
assert "tenferro_eager" in args, args
assert "libtorch_cpu" in args, args
assert "pytorch_cpu" in args, args
assert "jax_cpu" in args, args
print("#### Strategy: opt_flops")
print("")
print(f"Median ± IQR (ms). OMP_NUM_THREADS={threads}, RAYON_NUM_THREADS={threads}.")
print("")
print("| Instance | tenferro trace (ms) | tenferro eager (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |")
print("|---|---:|---:|---:|---:|---:|")
print("| bin_matmul_256 | 1.0 ± 0.1 | 1.2 ± 0.1 | 0.8 ± 0.1 | 0.7 ± 0.1 | **0.5 ± 0.1** |")
PY

cat > "$TMP/scripts/format_cpu_ops_results.py" <<'PY'
#!/usr/bin/env python3
import sys
assert len(sys.argv) == 2 and sys.argv[1].endswith('.csv'), sys.argv
threads = "4" if "_t4_" in sys.argv[1] else "1"
print("## PR884 CPU Benchmark Items")
print("")
print("Median ± IQR (ms).")
print("")
print("| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |")
print("|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
print(f"| matmul | `f64_square` | f64 | {threads} | `2x2` | 1.000 ± 0.100 | 1.100 ± 0.100 | 0.900 ± 0.100 | 0.800 ± 0.100 | **0.700 ± 0.100** |")
PY

chmod +x "$TMP/scripts/"*.sh "$TMP/scripts/format_results.py" "$TMP/scripts/format_cpu_ops_results.py"

(
  cd "$TMP"
  PATH="/usr/bin:/bin" ./scripts/run_all.sh 1 >/tmp/run_all_docs_test.out
  PATH="/usr/bin:/bin" ./scripts/run_all.sh 4 >>/tmp/run_all_docs_test.out
)

test -s "$TMP/data/results/results_t1_"*.md
test -s "$TMP/data/results/results_t4_"*.md
test -s "$TMP/data/results/cpu_ops_t1_"*.csv
test -s "$TMP/data/results/cpu_ops_t4_"*.csv
test -s "$TMP/data/results/cpu_ops_t1_"*.md
test -s "$TMP/data/results/cpu_ops_t4_"*.md
test -s "$TMP/result/einsum-results.md"
test -s "$TMP/result/cpu-benchmark-results.md"
test ! -e "$TMP/docs/results-einsum.md"
test ! -e "$TMP/docs/cpu-benchmark-results.md"
test -s "$TMP/setup_extern_called"

grep -q "Strategy: opt_flops" "$TMP/result/einsum-results.md"
grep -q "PyTorch Python" "$TMP/result/einsum-results.md"
grep -q "Torch C++" "$TMP/result/einsum-results.md"
grep -Fq "tenferro-rs commit: \`$TENFERRO_COMMIT\`" "$TMP/result/einsum-results.md"
grep -q "Latest run: \`./scripts/run_all.sh 4\`" "$TMP/result/einsum-results.md"
grep -q "## Threads: 1" "$TMP/result/einsum-results.md"
grep -q "## Threads: 4" "$TMP/result/einsum-results.md"
grep -q "data/results/results_t1_" "$TMP/result/einsum-results.md"
grep -q "data/results/results_t4_" "$TMP/result/einsum-results.md"
grep -q "CPU Information" "$TMP/result/einsum-results.md"
grep -q "Model:" "$TMP/result/einsum-results.md"

grep -q "PR884 CPU Benchmark Items" "$TMP/result/cpu-benchmark-results.md"
grep -q "tenferro-rs eager mode" "$TMP/result/cpu-benchmark-results.md"
grep -q "tenferro-rs trace mode" "$TMP/result/cpu-benchmark-results.md"
grep -q "PyTorch Python" "$TMP/result/cpu-benchmark-results.md"
grep -Fq "tenferro-rs commit: \`$TENFERRO_COMMIT\`" "$TMP/result/cpu-benchmark-results.md"
grep -q "Latest run: \`./scripts/run_all.sh 4\`" "$TMP/result/cpu-benchmark-results.md"
grep -q "## Threads: 1" "$TMP/result/cpu-benchmark-results.md"
grep -q "## Threads: 4" "$TMP/result/cpu-benchmark-results.md"
grep -q "data/results/cpu_ops_t1_" "$TMP/result/cpu-benchmark-results.md"
grep -q "data/results/cpu_ops_t4_" "$TMP/result/cpu-benchmark-results.md"
grep -q "CPU Information" "$TMP/result/cpu-benchmark-results.md"
grep -q "Logical CPUs:" "$TMP/result/cpu-benchmark-results.md"
