#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

grep -q 'path = "extern/tenferro-rs/crates/tenferro-ad"' "$ROOT/Cargo.toml"
grep -q 'path = "extern/tenferro-rs/crates/tenferro-einsum"' "$ROOT/Cargo.toml"
grep -q 'path = "extern/tenferro-rs/crates/tenferro-linalg"' "$ROOT/Cargo.toml"
grep -q 'path = "extern/tenferro-rs/crates/tenferro-runtime"' "$ROOT/Cargo.toml"
grep -q 'path = "extern/tenferro-rs/crates/tenferro-tensor"' "$ROOT/Cargo.toml"
