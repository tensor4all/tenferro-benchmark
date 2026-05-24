#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

grep -q 'path = "extern/tenferro-rs/tenferro"' "$ROOT/Cargo.toml"
grep -q 'path = "extern/tenferro-rs/tenferro-einsum"' "$ROOT/Cargo.toml"
grep -q 'path = "extern/tenferro-rs/tenferro-tensor"' "$ROOT/Cargo.toml"
grep -q 'path = "extern/tenferro-rs/tenferro-ops"' "$ROOT/Cargo.toml"
grep -q 'path = "extern/tenferro-rs/tenferro-device"' "$ROOT/Cargo.toml"
