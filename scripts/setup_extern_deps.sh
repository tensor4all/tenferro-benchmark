#!/usr/bin/env bash
set -euo pipefail

# Prepare repo-local external dependencies used by the benchmark suite.
# This script is intended to be sourced by run_all.sh so exported variables
# remain available to child benchmark scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXTERN_DIR="$PROJECT_DIR/extern"

# shellcheck source=scripts/cpu_blas_provider.sh
source "$SCRIPT_DIR/cpu_blas_provider.sh"

TENFERRO_REPO_URL="${TENFERRO_REPO_URL:-https://github.com/tensor4all/tenferro-rs.git}"
TENFERRO_REF="${TENFERRO_REF:-main}"
TENFERRO_UPDATE="${TENFERRO_UPDATE:-0}"
SETUP_EXTERN_MIGRATE_SIBLINGS="${SETUP_EXTERN_MIGRATE_SIBLINGS:-0}"

TENFERRO_DIR="${TENFERRO_RS_DIR:-$EXTERN_DIR/tenferro-rs}"

log() {
    printf '[setup_extern_deps] %s\n' "$*" >&2
}

ensure_openblas_root() {
    if [[ -n "${OPENBLAS_ROOT:-}" ]]; then
        export OPENBLAS_ROOT
        return
    fi
    if command -v brew >/dev/null 2>&1 && brew --prefix openblas >/dev/null 2>&1; then
        export OPENBLAS_ROOT="$(brew --prefix openblas)"
        return
    fi
    cat >&2 <<'EOF'
OPENBLAS_ROOT is required.
Set OPENBLAS_ROOT to the OpenBLAS prefix, or install OpenBLAS with Homebrew.
EOF
    exit 1
}

require_clean_checkout() {
    local name="$1"
    local dest="$2"

    if [[ -n "$(git -C "$dest" status --porcelain)" ]]; then
        cat >&2 <<EOF
$name has local changes at $dest.
Commit, stash, or remove those changes before explicitly updating the checkout.
The default setup path reuses the existing checkout, including dirty changes.
EOF
        return 1
    fi
}

resolve_git_ref() {
    local name="$1"
    local dest="$2"
    local ref="$3"

    git -C "$dest" fetch --tags origin
    if git -C "$dest" rev-parse --verify --quiet "origin/$ref^{commit}" >/dev/null; then
        printf 'origin/%s\n' "$ref"
        return
    fi
    if git -C "$dest" rev-parse --verify --quiet "$ref^{commit}" >/dev/null; then
        printf '%s\n' "$ref"
        return
    fi
    if git -C "$dest" fetch origin "$ref" >/dev/null 2>&1; then
        printf 'FETCH_HEAD\n'
        return
    fi

    echo "Could not resolve $name ref '$ref' in $dest" >&2
    return 1
}

checkout_git_ref() {
    local name="$1"
    local dest="$2"
    local ref="$3"
    local target

    [[ -n "$ref" ]] || return
    require_clean_checkout "$name" "$dest" || return
    target="$(resolve_git_ref "$name" "$dest" "$ref")" || return
    log "updating $name to $ref"
    git -C "$dest" checkout -q --detach "$target"
    git -C "$dest" submodule update --init --recursive
    log "$name commit $(git -C "$dest" rev-parse --short HEAD)"
}

clone_or_update_tenferro_checkout() {
    local name="tenferro-rs"
    local dest="$TENFERRO_DIR"
    local sibling="$PROJECT_DIR/../$name"

    mkdir -p "$(dirname "$dest")"
    if [[ -d "$dest/.git" ]]; then
        log "$name already exists at ${dest#$PROJECT_DIR/}; reusing current checkout"
        if [[ "$TENFERRO_UPDATE" == "1" ]]; then
            checkout_git_ref "$name" "$dest" "$TENFERRO_REF"
        fi
        return
    fi
    if [[ -e "$dest" ]]; then
        cat >&2 <<EOF
$name exists at $dest, but it is not a git checkout.
Remove it or set TENFERRO_RS_DIR to a valid tenferro-rs checkout.
EOF
        return 1
    fi

    if [[ "$SETUP_EXTERN_MIGRATE_SIBLINGS" == "1" && -d "$sibling/.git" ]]; then
        log "moving existing ../$name into extern/$name"
        mv "$sibling" "$dest"
        if [[ "$TENFERRO_UPDATE" == "1" ]]; then
            checkout_git_ref "$name" "$dest" "$TENFERRO_REF"
        fi
        return
    fi

    log "cloning $name into ${dest#$PROJECT_DIR/}"
    git clone --recursive "$TENFERRO_REPO_URL" "$dest"
    if [[ "$TENFERRO_UPDATE" == "1" ]]; then
        checkout_git_ref "$name" "$dest" "$TENFERRO_REF"
    else
        log "$name commit $(git -C "$dest" rev-parse --short HEAD)"
    fi
}

ensure_tenferro_checkout() {
    clone_or_update_tenferro_checkout
}

main() {
    TENFERRO_CPU_FEATURES="$(normalize_cpu_blas_features "${TENFERRO_CPU_FEATURES:-}")"
    export TENFERRO_CPU_FEATURES
    case "$TENFERRO_CPU_FEATURES" in
        system-openblas)
            ensure_openblas_root
            ensure_blas_env_for_features "$TENFERRO_CPU_FEATURES"
            ;;
        system-mkl)
            ensure_blas_env_for_features "$TENFERRO_CPU_FEATURES"
            ;;
    esac
    ensure_tenferro_checkout

    export TENFERRO_RS_DIR="$TENFERRO_DIR"

    [[ -n "${OPENBLAS_ROOT:-}" ]] && log "OPENBLAS_ROOT=$OPENBLAS_ROOT"
    [[ -n "${MKLROOT:-}" ]] && log "MKLROOT=$MKLROOT"
    log "TENFERRO_RS_DIR=$TENFERRO_RS_DIR"
    log "TENFERRO_CPU_FEATURES=$TENFERRO_CPU_FEATURES"
    return 0
}

if [[ "${SETUP_EXTERN_DEPS_SKIP_MAIN:-0}" != "1" ]]; then
    main "$@"
fi
