use std::path::PathBuf;

/// Probe the tenferro-rs source this harness builds against for APIs that
/// differ between the #1946 audit baseline (5a4e7fd84) and its repair, so the
/// cross-revision route-diagnostic and session-matrix binaries compile on both.
/// A probe only enables a code path; an absent API makes the affected cases
/// report `unsupported`, never success.
///
/// Cargo does not notice a re-pointed `extern/tenferro-rs` symlink reliably
/// (file mtimes can go backwards), so build each tenferro-rs revision into
/// its own `CARGO_TARGET_DIR` (scripts/build_for_tenferro_rev.sh does that).
fn probe_tenferro_apis() {
    let root =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("extern/tenferro-rs");
    let probes = [(
        "tenferro_session_einsum",
        "crates/tenferro-einsum/src/eager_ad.rs",
        "pub trait EagerSessionEinsumExt",
    )];
    for (cfg, file, needle) in probes {
        println!("cargo:rustc-check-cfg=cfg({cfg})");
        let path = root.join(file);
        println!("cargo:rerun-if-changed={}", path.display());
        if std::fs::read_to_string(&path).is_ok_and(|text| text.contains(needle)) {
            println!("cargo:rustc-cfg={cfg}");
        }
    }
    if let Ok(target) = std::fs::canonicalize(&root) {
        println!(
            "cargo:rustc-env=TENFERRO_RS_BUILD_PATH={}",
            target.display()
        );
    }
}

fn main() {
    // tenferro-rs 0.12 has one CPU backend, so this build no longer selects or
    // links a vendor BLAS/LAPACK provider (tenferro-rs #2608). Nothing here
    // reads OPENBLAS_ROOT or MKLROOT any more.
    probe_tenferro_apis();
}
