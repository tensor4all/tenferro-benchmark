#!/usr/bin/env bash
# Revision builds must preserve both ordinary checkouts and existing symlinks.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -r -- "$TMP"' EXIT
mkdir -p "$TMP/scripts" "$TMP/bin" "$TMP/extern/tenferro-rs/crates/tenferro-cpu"
cp "$ROOT/scripts/build_for_tenferro_rev.sh" "$ROOT/scripts/cpu_blas_provider.sh" "$TMP/scripts/"
printf 'preserve this checkout\n' > "$TMP/extern/tenferro-rs/crates/tenferro-cpu/marker"
git -C "$TMP/extern/tenferro-rs" init -q
git -C "$TMP/extern/tenferro-rs" add .
git -C "$TMP/extern/tenferro-rs" -c user.name=Fixture -c user.email=fixture@example.invalid commit -qm fixture
cat > "$TMP/bin/cargo" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ "$(cd extern/tenferro-rs && pwd -P)" == "$EXPECTED_TENFERRO_DIR" ]]
grep -qx 'preserve this checkout' extern/tenferro-rs/crates/tenferro-cpu/marker
[[ "$CARGO_TARGET_DIR" == "$PWD/target/tenferro-rev/"* ]]
SH
chmod +x "$TMP/bin/cargo"
export PATH="$TMP/bin:$PATH"
export TENFERRO_CPU_FEATURES=native
cd "$TMP"
export EXPECTED_TENFERRO_DIR="$TMP/extern/tenferro-rs"
bash scripts/build_for_tenferro_rev.sh "$EXPECTED_TENFERRO_DIR" "$TMP/build.env" benchmark_cpu_session
[[ -d extern/tenferro-rs/.git && ! -L extern/tenferro-rs ]]
[[ -z "$(git -C extern/tenferro-rs status --porcelain)" ]]

# A different revision must not replace a real checkout.
cp -a extern/tenferro-rs alternate
if bash scripts/build_for_tenferro_rev.sh "$TMP/alternate" "$TMP/other.env" benchmark_cpu_session > "$TMP/reject.log" 2>&1; then
    echo 'different checkout replaced a real directory' >&2
    exit 1
fi
[[ -d extern/tenferro-rs/.git && ! -L extern/tenferro-rs ]]
[[ -z "$(git -C extern/tenferro-rs status --porcelain)" ]]

# Existing revision-switching users retain their original link after a build.
mv extern/tenferro-rs original
ln -s "$TMP/original" extern/tenferro-rs
export EXPECTED_TENFERRO_DIR="$TMP/alternate"
bash scripts/build_for_tenferro_rev.sh "$EXPECTED_TENFERRO_DIR" "$TMP/other.env" benchmark_cpu_session
[[ "$(readlink extern/tenferro-rs)" == "$TMP/original" ]]
[[ -z "$(git -C original status --porcelain)" ]]
echo 'revision build checkout preservation OK'
