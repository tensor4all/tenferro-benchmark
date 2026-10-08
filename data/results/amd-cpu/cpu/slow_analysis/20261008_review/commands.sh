#!/usr/bin/env bash
# Executed sequentially inside the Linux CPU devcontainer.
export CARGO_TARGET_DIR="$PWD/target/compatibility-mkl" CARGO_BUILD_JOBS=8
export LD_LIBRARY_PATH="$MKLROOT/lib:/opt/intel/oneapi/compiler/2026.1/lib:${LD_LIBRARY_PATH:-}"
cargo build --release --no-default-features --features system-mkl --example cpu_gap_mwe
.venv/bin/python scripts/confirm_cpu_gaps.py data/results/amd-cpu/cpu/slow_analysis/20261008_issues
# Added patterned IFFT, cast and reshape controls at harness 672df86.
.venv/bin/python scripts/confirm_cpu_gaps.py data/results/amd-cpu/cpu/slow_analysis/20261008_extended ifft-pattern cast reshape
# Matched logical reshape output/layout at harness 1b0e3aa; supersedes earlier reshape.
.venv/bin/python scripts/confirm_cpu_gaps.py data/results/amd-cpu/cpu/slow_analysis/20261008_reshape_matched reshape
# configure_cpu_thread_env 1/4 and idle guard run before EVERY timed process.
# GELU and softmax screening/reproduction at harness 6d06ca0.
.venv/bin/python scripts/confirm_cpu_gaps.py data/results/amd-cpu/cpu/slow_analysis/20261008_activation_softmax gelu softmax
# Initial GELU noise gate failed: longer intervals, same thresholds; harness 578d84e.
CPU_GAP_TARGET_NS=10000000 .venv/bin/python scripts/confirm_cpu_gaps.py data/results/amd-cpu/cpu/slow_analysis/20261008_gelu_10ms gelu
# Match logical GELU [1024,64] in native PyTorch layout at harness 70ffcd4.
CPU_GAP_TARGET_NS=10000000 .venv/bin/python scripts/confirm_cpu_gaps.py data/results/amd-cpu/cpu/slow_analysis/20261008_gelu_shape_matched gelu
