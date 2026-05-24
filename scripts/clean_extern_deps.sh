#!/usr/bin/env bash
set -euo pipefail

# Remove repo-local external dependency checkouts created by setup_extern_deps.sh.
# Only the known dependency directories are removed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXTERN_DIR="$PROJECT_DIR/extern"

log() {
    printf '[clean_extern_deps] %s\n' "$*" >&2
}

remove_checkout() {
    local name="$1"
    local path="$EXTERN_DIR/$name"

    if [[ ! -e "$path" ]]; then
        log "$name not found at ${path#$PROJECT_DIR/}"
        return
    fi

    case "$path" in
        "$PROJECT_DIR"/extern/tenferro-rs|"$PROJECT_DIR"/extern/pytorch-openblas)
            log "removing ${path#$PROJECT_DIR/}"
            rm -rf "$path"
            ;;
        *)
            echo "Refusing to remove unexpected path: $path" >&2
            exit 1
            ;;
    esac
}

remove_checkout tenferro-rs
remove_checkout pytorch-openblas

if [[ -d "$EXTERN_DIR" ]] && ! find "$EXTERN_DIR" -mindepth 1 -maxdepth 1 | grep -q .; then
    log "removing empty extern directory"
    rmdir "$EXTERN_DIR"
fi
