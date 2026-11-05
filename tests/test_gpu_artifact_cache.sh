#!/usr/bin/env bash
# The artifact-cache discovery in scripts/run_gpu_public_api.sh must never take the
# runner down: it runs under set -euo pipefail on hosts whose driver may not answer
# compute_cap and whose toolkit may keep libnvrtc somewhere else.
set -euo pipefail

cd "$(dirname "$0")/.."
block="$(mktemp)"
trap 'rm -f "$block"' EXIT
# The cache block is the only part of the runner that is pure discovery, so it can
# be exercised without a GPU, a fixture set or a CUDA build.
sed -n '/^# Each family starts two Rust arms/,/^echo "CUDA artifacts/p' \
    scripts/run_gpu_public_api.sh | grep -v '^echo "CUDA artifacts' > "$block"

run_block() {
    GPU_BENCH_PHYSICAL_DEVICE=0 bash -c '
        set -euo pipefail
        source "$1"
        printf "%s\n" "$TENFERRO_CUDA_ARTIFACT_CACHE"
    ' _ "$block"
}

fakes="$(mktemp -d)"
trap 'rm -f "$block"; rm -rf "$fakes"' EXIT
printf '#!/usr/bin/env bash\nexit 1\n' > "$fakes/nvidia-smi"
chmod +x "$fakes/nvidia-smi"

# A toolkit without libnvrtc and a driver that rejects the compute_cap query must
# still yield a usable directory name instead of aborting the run.
[ "$(PATH="$fakes:$PATH" CUDA_HOME=/nonexistent run_block)" = "${XDG_CACHE_HOME:-$HOME/.cache}/tenferro-benchmark/cuda-artifacts/unknown/unknown" ]

# An explicit value wins, including the memory-only one, and skips discovery.
[ "$(PATH="$fakes:$PATH" CUDA_HOME=/nonexistent TENFERRO_CUDA_ARTIFACT_CACHE=none run_block)" = "none" ]

# The key carries both the NVRTC version and the device capability when they exist.
toolkit="$fakes/cuda"
mkdir -p "$toolkit/lib64"
: > "$toolkit/lib64/libnvrtc.so.12.6.77"
mkdir -p "$fakes/bin"
cat > "$fakes/bin/smi-ok" <<'EOS'
#!/usr/bin/env bash
printf '8.0\n'
EOS
chmod +x "$fakes/bin/smi-ok"
mkdir -p "$fakes/bin2"
cp "$fakes/bin/smi-ok" "$fakes/bin2/nvidia-smi"
[ "$(PATH="$fakes/bin2:$PATH" CUDA_HOME="$toolkit" run_block)" = "${XDG_CACHE_HOME:-$HOME/.cache}/tenferro-benchmark/cuda-artifacts/12.6.77/sm_80" ]

echo "gpu artifact cache ok"
