#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

grep -q 'path = "extern/tenferro-rs/tenferro-ad"' "$ROOT/Cargo.toml"
grep -q 'path = "extern/tenferro-rs/tenferro-einsum"' "$ROOT/Cargo.toml"
grep -q 'path = "extern/tenferro-rs/tenferro-linalg"' "$ROOT/Cargo.toml"
grep -q 'path = "extern/tenferro-rs/tenferro-runtime"' "$ROOT/Cargo.toml"
grep -q 'path = "extern/tenferro-rs/tenferro-tensor"' "$ROOT/Cargo.toml"
