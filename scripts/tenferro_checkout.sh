#!/usr/bin/env bash

# Shared identity of the tenferro-rs checkout this harness builds against.
#
# Every tenferro-rs dependency in Cargo.toml is a path dependency on
# extern/tenferro-rs, so collectors describe that checkout and never the
# TENFERRO_RS_DIR clone location, which Cargo does not read.

tenferro_checkout_commit() {
    local checkout="${1:?checkout required}"
    # A normal clone has a .git directory; a linked worktree has a .git file.
    [[ -e "$checkout/.git" ]] || return 0
    command -v git >/dev/null 2>&1 || return 0
    git -C "$checkout" rev-parse HEAD 2>/dev/null || true
}

update_tenferro_checkout_if_main() {
    local checkout="${1:?checkout required}"
    # Offline compute nodes cannot reach GitHub; they keep the checked-out
    # revision and still record it through tenferro_checkout_commit.
    [[ "${GPU_BENCH_NO_PULL:-0}" == 1 ]] && return 0
    [[ -e "$checkout/.git" ]] || return 0
    command -v git >/dev/null 2>&1 || return 0
    [[ "$(git -C "$checkout" branch --show-current 2>/dev/null)" == main ]] || return 0
    # GPU images have git but no SSH client; public GitHub fetches need no key.
    git -C "$checkout" -c 'url.https://github.com/.insteadOf=git@github.com:' pull --ff-only
}
