resolve_git_commit() {
    local checkout_dir="$1"
    if [[ -d "$checkout_dir/.git" ]] && command -v git >/dev/null 2>&1; then
        git -C "$checkout_dir" rev-parse HEAD 2>/dev/null || true
    fi
}

write_cpu_info_section() {
    if command -v uv >/dev/null 2>&1; then
        uv run python "$PROJECT_DIR/scripts/collect_cpu_info.py" --markdown \
            || python3 "$PROJECT_DIR/scripts/collect_cpu_info.py" --markdown
    else
        python3 "$PROJECT_DIR/scripts/collect_cpu_info.py" --markdown
    fi
}

write_thread_env_section() {
    echo "## Thread Environment"
    echo ""
    for key in \
        OMP_NUM_THREADS \
        OMP_THREAD_LIMIT \
        OMP_DYNAMIC \
        RAYON_NUM_THREADS \
        OPENBLAS_NUM_THREADS \
        GOTO_NUM_THREADS \
        MKL_NUM_THREADS \
        VECLIB_MAXIMUM_THREADS \
        VECLIB_NUM_THREADS \
        NUMEXPR_NUM_THREADS \
        BLIS_NUM_THREADS \
        XLA_FLAGS; do
        echo "- ${key}: \`${!key:-}\`"
    done
}

read_run_yaml_section_scalar() {
    local run_yaml="$1"
    local section="$2"
    local key="$3"
    awk -v section="$section" -v key="$key" '
        $0 == section ":" {
            in_section = 1
            next
        }
        in_section && $0 ~ /^[^[:space:]]/ {
            exit
        }
        in_section {
            line = $0
            sub(/^[[:space:]]+/, "", line)
            if (line ~ "^" key ":[[:space:]]*") {
                sub("^" key ":[[:space:]]*", "", line)
                gsub(/^'\''|'\''$/, "", line)
                gsub(/^"|"$/, "", line)
                print line
                exit
            }
        }
    ' "$run_yaml"
}

write_blas_backend_section() {
    local run_yaml="$1"
    local implementation version root library
    implementation="$(read_run_yaml_section_scalar "$run_yaml" "blas" "implementation")"
    version="$(read_run_yaml_section_scalar "$run_yaml" "blas" "version")"
    root="$(read_run_yaml_section_scalar "$run_yaml" "blas" "root")"
    library="$(read_run_yaml_section_scalar "$run_yaml" "blas" "library")"

    echo "## Tenferro CPU BLAS Backend"
    echo ""
    echo "- tenferro-rs features: \`$TENFERRO_CPU_FEATURES\`"
    echo "- TENFERRO_CPU_BACKEND_KIND: \`${TENFERRO_CPU_BACKEND_KIND:-}\`"
    [[ -n "$implementation" ]] && echo "- BLAS implementation: \`$implementation\`"
    [[ -n "$version" ]] && echo "- BLAS version: \`$version\`"
    [[ -n "$root" ]] && echo "- BLAS root: \`$root\`"
    [[ -n "$library" ]] && echo "- BLAS library: \`$library\`"
}

write_python_backend_section() {
    local run_yaml="$1"
    echo "## Python Backend Providers"
    echo ""
    run_python_script - "$run_yaml" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:  # noqa: BLE001
    print(f"- unavailable: PyYAML is not available ({exc})")
    raise SystemExit(0)

run_path = Path(sys.argv[1])
try:
    run = yaml.safe_load(run_path.read_text())
except Exception as exc:  # noqa: BLE001
    print(f"- unavailable: failed to read run metadata ({exc})")
    raise SystemExit(0)

backends = (run or {}).get("python_backends") or {}
if not backends:
    print("- unavailable: provider metadata was not collected")
    raise SystemExit(0)

labels = {"pytorch": "PyTorch", "jax": "JAX"}
for key in ("pytorch", "jax"):
    info = backends.get(key) or {}
    label = labels[key]
    if not info.get("available"):
        reason = info.get("reason") or "unavailable"
        print(f"- {label}: unavailable ({reason})")
        continue
    provider = info.get("provider") or "unknown"
    version = info.get("version") or "unknown"
    if key == "jax":
        dot_backend = info.get("dot_backend") or provider
        details = [f"dot backend `{dot_backend}`", f"version `{version}`"]
        if info.get("jaxlib_version"):
            details.append(f"jaxlib `{info['jaxlib_version']}`")
        if info.get("backend"):
            details.append(f"default backend `{info['backend']}`")
        if info.get("lapack_provider"):
            details.append(f"LAPACK provider `{info['lapack_provider']}`")
    else:
        details = [f"BLAS provider `{provider}`", f"version `{version}`"]
        if info.get("blas_info"):
            details.append(f"BLAS_INFO `{info['blas_info']}`")
        if info.get("lapack_info"):
            details.append(f"LAPACK_INFO `{info['lapack_info']}`")
        if info.get("backend"):
            details.append(f"backend `{info['backend']}`")
    print(f"- {label}: " + ", ".join(details))
    deps = info.get("linked_libraries") or []
    if deps:
        dep_label = "linked LAPACK libs" if key == "jax" else "linked BLAS/LAPACK libs"
        print(f"  - {dep_label}: " + "; ".join(f"`{dep}`" for dep in deps))
PY
}

write_julia_backend_section() {
    local run_yaml="$1"
    echo "## Julia / OMEinsum.jl Backend"
    echo ""
    run_python_script - "$run_yaml" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:  # noqa: BLE001
    print(f"- unavailable: PyYAML is not available ({exc})")
    raise SystemExit(0)

run_path = Path(sys.argv[1])
try:
    run = yaml.safe_load(run_path.read_text())
except Exception as exc:  # noqa: BLE001
    print(f"- unavailable: failed to read run metadata ({exc})")
    raise SystemExit(0)

julia = (run or {}).get("julia") or {}
if not julia.get("available"):
    reason = julia.get("reason") or "unavailable"
    print(f"- Julia: unavailable ({reason})")
    raise SystemExit(0)

details = [f"version `{julia.get('version') or 'unknown'}`"]
if julia.get("omeinsum_version"):
    details.append(f"OMEinsum.jl `{julia['omeinsum_version']}`")
if julia.get("blas_provider"):
    details.append(f"BLAS provider `{julia['blas_provider']}`")
if julia.get("threads") is not None:
    details.append(f"probe threads `{julia['threads']}`")
print("- Julia: " + ", ".join(details))
PY
    echo ""
    echo "- \`omeinsum-jl\` (mode \`omeinsum_path\` in the log/report) always"
    echo "  executes the instance's precomputed \`opt_flops\`/\`opt_size\` path via"
    echo "  \`OMEinsum.DynamicEinCode\` pairwise contractions; OMEinsum's own"
    echo "  contraction-order optimizer is never invoked, so the comparison"
    echo "  against tenferro/PyTorch/JAX (which also use the precomputed path)"
    echo "  stays fair."
    echo "- \`JULIA_NUM_THREADS\` also pins \`LinearAlgebra.BLAS.set_num_threads\`,"
    echo "  matching the BLAS thread pinning used by the other CPU backends."
    echo "- Julia is column-major like tenferro-rs, so the einsum runner uses"
    echo "  \`format_string_colmajor\` / \`shapes_colmajor\` directly with no"
    echo "  PyTorch/JAX-style layout reconstruction."
    echo "- OMEinsum dispatches its pairwise contractions to Julia's own BLAS"
    echo "  (libblastrampoline, OpenBLAS by default; the provider that ran is"
    echo "  reported above). When that differs from the provider tenferro-rs and"
    echo "  PyTorch link against, matmul-shaped rows partly compare BLAS"
    echo "  implementations rather than einsum-runtime overhead; read them"
    echo "  together with the recorded providers. \`docs/einsum-suite.md\` records"
    echo "  the measured OMEinsum-over-its-own-BLAS overhead."
}

run_python_script() {
    local script="$1"
    shift
    if command -v uv >/dev/null 2>&1; then
        uv run python "$script" "$@" || python3 "$script" "$@"
    else
        python3 "$script" "$@"
    fi
}

resolve_suite_instance_ids() {
    local suite_file="$1"
    run_python_script "$PROJECT_DIR/scripts/suite_instances.py" --suite-file "$suite_file"
}

write_run_metadata() {
    local run_yaml="$1"
    local timestamp="$2"
    local tenferro_dir="${TENFERRO_RS_DIR:-$PROJECT_DIR/extern/tenferro-rs}"
    local blas_impl
    blas_impl="$(blas_impl_for_features "$TENFERRO_CPU_FEATURES")"
    local args=(
        --suite-id "$CPU_SUITE_ID"
        --target-profile "$BENCHMARK_TARGET_PROFILE"
        --suite-file "${CPU_SUITE_FILE#$PROJECT_DIR/}"
        --timestamp "$timestamp"
        --tenferro-dir "$tenferro_dir"
        --features "$TENFERRO_CPU_FEATURES"
        --blas "$blas_impl"
        --output "$run_yaml"
    )
    if [[ -n "$TENFERRO_COMMIT" ]]; then
        args+=(--tenferro-commit "$TENFERRO_COMMIT")
    fi
    run_python_script "$PROJECT_DIR/scripts/collect_run_metadata.py" "${args[@]}"
}

write_einsum_report() {
    local report="$1"
    mkdir -p "$(dirname "$report")"
    {
        echo "# Einsum Benchmark Results"
        echo ""
        echo "- Suite: \`$CPU_SUITE_ID\`"
        echo "- Target profile: \`$BENCHMARK_TARGET_PROFILE\`"
        echo "- Suite file: \`${CPU_SUITE_FILE#$PROJECT_DIR/}\`"
        echo "- Run metadata: \`${CPU_RUN_YAML#$PROJECT_DIR/}\`"
        echo "- Timestamp: \`$BENCHMARK_TIMESTAMP\`"
        echo ""
        echo "Latest run: \`./scripts/run_all.sh $NUM_THREADS\`."
        echo ""
        echo "This file is generated from one suite run under \`${CPU_RUN_DIR#$PROJECT_DIR/}\`."
        echo ""
        if [[ -n "$TENFERRO_COMMIT" ]]; then
            echo "- tenferro-rs commit: \`$TENFERRO_COMMIT\`"
            echo ""
        fi
        write_cpu_info_section
        echo ""
        write_thread_env_section
        echo ""
        write_blas_backend_section "$CPU_RUN_YAML"
        echo ""
        write_python_backend_section "$CPU_RUN_YAML"
        echo ""
        write_julia_backend_section "$CPU_RUN_YAML"
        echo ""

        echo "## Threads: $NUM_THREADS"
        echo ""
        echo "- Source table: \`${MARKDOWN_TABLE#$PROJECT_DIR/}\`"
        echo ""
        echo "Logs:"
        echo ""
        for log in \
            "$TENFERRO_TRACE_LOG" \
            "$TENFERRO_EAGER_LOG" \
            "$PYTORCH_LOG" \
            "$JAX_LOG" \
            "$JULIA_LOG"; do
            [[ -f "$log" && -s "$log" ]] && echo "- \`${log#$PROJECT_DIR/}\`"
        done
        echo ""
        cat "$MARKDOWN_TABLE"
    } > "$report"
}

write_linalg_ad_report() {
    local report="$1"
    mkdir -p "$(dirname "$report")"
    {
        echo "# CPU Linalg JVP/VJP Benchmark Results"
        echo ""
        echo "- Suite: \`cpu/linalg_jvp_vjp\`"
        echo "- Target profile: \`$BENCHMARK_TARGET_PROFILE\`"
        echo "- Timestamp: \`$BENCHMARK_TIMESTAMP\`"
        echo ""
        echo "Latest run: \`./scripts/run_all.sh $NUM_THREADS\`."
        echo ""
        echo "Derived from the CPU ops CSV under \`${CPU_RUN_DIR#$PROJECT_DIR/}\`."
        echo ""
        if [[ -n "$TENFERRO_COMMIT" ]]; then
            echo "- tenferro-rs commit: \`$TENFERRO_COMMIT\`"
            echo ""
        fi
        write_cpu_info_section
        echo ""
        write_thread_env_section
        echo ""
        write_blas_backend_section "$CPU_RUN_YAML"
        echo ""
        write_python_backend_section "$CPU_RUN_YAML"
        echo ""
        echo "## Threads: $NUM_THREADS"
        echo ""
        [[ -f "$CPU_OPS_LOG" ]] && echo "- CSV: \`${CPU_OPS_LOG#$PROJECT_DIR/}\`"
        echo "- Source table: \`${LINALG_AD_MD#$PROJECT_DIR/}\`"
        echo ""
        cat "$LINALG_AD_MD"
    } > "$report"
}

write_cpu_report() {
    local report="$1"
    mkdir -p "$(dirname "$report")"
    {
        echo "# CPU Benchmark Results"
        echo ""
        echo "- Suite: \`cpu/cpu_ops\`"
        echo "- Target profile: \`$BENCHMARK_TARGET_PROFILE\`"
        echo "- Timestamp: \`$BENCHMARK_TIMESTAMP\`"
        echo ""
        echo "Latest run: \`./scripts/run_all.sh $NUM_THREADS\`."
        echo ""
        echo "This file is generated from one CPU ops run under \`${CPU_RUN_DIR#$PROJECT_DIR/}\`."
        echo ""
        if [[ -n "$TENFERRO_COMMIT" ]]; then
            echo "- tenferro-rs commit: \`$TENFERRO_COMMIT\`"
            echo ""
        fi
        write_cpu_info_section
        echo ""
        write_thread_env_section
        echo ""
        write_blas_backend_section "$CPU_RUN_YAML"
        echo ""
        write_python_backend_section "$CPU_RUN_YAML"
        echo ""
        echo "## Threads: $NUM_THREADS"
        echo ""
        [[ -f "$CPU_OPS_LOG" ]] && echo "- CSV: \`${CPU_OPS_LOG#$PROJECT_DIR/}\`"
        echo "- Source table: \`${CPU_OPS_MD#$PROJECT_DIR/}\`"
        echo ""
        cat "$CPU_OPS_MD"
    } > "$report"
}
