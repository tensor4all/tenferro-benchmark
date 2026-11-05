#!/usr/bin/env bash
# The artifact-cache discovery in scripts/gpu_preflight.sh must never take the
# runner down: every collector calls it under set -euo pipefail on hosts whose
# driver may not answer compute_cap and whose toolkit may keep libnvrtc elsewhere.
set -euo pipefail

cd "$(dirname "$0")/.."

run_discovery() {
    GPU_BENCH_PHYSICAL_DEVICE=0 bash -c '
        set -euo pipefail
        source scripts/gpu_preflight.sh
        configure_cuda_artifact_cache >/dev/null
        printf "%s\n" "$TENFERRO_CUDA_ARTIFACT_CACHE"
    '
}

fakes="$(mktemp -d)"
trap 'rm -rf "$fakes"' EXIT
printf '#!/usr/bin/env bash\nexit 1\n' > "$fakes/nvidia-smi"
chmod +x "$fakes/nvidia-smi"

# A toolkit without libnvrtc and a driver that rejects the compute_cap query must
# still yield a usable directory name instead of aborting the run.
[ "$(PATH="$fakes:$PATH" CUDA_HOME=/nonexistent run_discovery)" = "${XDG_CACHE_HOME:-$HOME/.cache}/tenferro-benchmark/cuda-artifacts/unknown/unknown" ]

# An explicit value wins, including the memory-only one, and skips discovery.
[ "$(PATH="$fakes:$PATH" CUDA_HOME=/nonexistent TENFERRO_CUDA_ARTIFACT_CACHE=none run_discovery)" = "none" ]

# Without HOME or XDG_CACHE_HOME there is no directory to name: keep the
# memory-only behaviour instead of aborting under set -u.
[ "$(env -u HOME -u XDG_CACHE_HOME PATH="$fakes:$PATH" CUDA_HOME=/nonexistent bash -c '
    set -u
    source scripts/gpu_preflight.sh
    configure_cuda_artifact_cache >/dev/null 2>&1
    printf "%s\n" "${TENFERRO_CUDA_ARTIFACT_CACHE:-unset}"
' | tr -d '\n')" = "unset" ]

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
[ "$(PATH="$fakes/bin2:$PATH" CUDA_HOME="$toolkit" run_discovery)" = "${XDG_CACHE_HOME:-$HOME/.cache}/tenferro-benchmark/cuda-artifacts/12.6.77/sm_80" ]

# configure_gpu_bench_env exports it too, so the collectors that use it do not
# need a second discovery of their own.
cat > "$fakes/bin2/nvidia-smi" <<'EOS'
#!/usr/bin/env bash
case "$*" in
    *--query-gpu=uuid*) printf 'GPU-00000000-0000-0000-0000-000000000000\n' ;;
    *) printf '8.0\n' ;;
esac
EOS
[ "$(PATH="$fakes/bin2:$PATH" CUDA_HOME="$toolkit" bash -c '
    set -euo pipefail
    source scripts/gpu_preflight.sh
    configure_gpu_bench_env 0 >/dev/null
    printf "%s\n" "$TENFERRO_CUDA_ARTIFACT_CACHE"
')" = "${XDG_CACHE_HOME:-$HOME/.cache}/tenferro-benchmark/cuda-artifacts/12.6.77/sm_80" ]

echo "gpu artifact cache ok"
