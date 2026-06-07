#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/scripts" "$TMP/extern/tenferro-rs" "$TMP/extern/keep-me"
cp "$ROOT/scripts/clean_extern_deps.sh" "$TMP/scripts/clean_extern_deps.sh"

touch "$TMP/extern/tenferro-rs/sentinel"
touch "$TMP/extern/keep-me/sentinel"

(
  cd "$TMP"
  ./scripts/clean_extern_deps.sh
)

test ! -e "$TMP/extern/tenferro-rs"
test -e "$TMP/extern/keep-me/sentinel"
