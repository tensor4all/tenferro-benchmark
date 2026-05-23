use std::env;
use std::path::PathBuf;

fn main() {
    if env::var_os("CARGO_FEATURE_SYSTEM_OPENBLAS").is_none() {
        return;
    }

    let root = env::var("OPENBLAS_ROOT")
        .expect("OPENBLAS_ROOT must be set when building with feature `system-openblas`");
    let root = PathBuf::from(root);
    let lib_dir = root.join("lib");
    let include_dir = root.join("include");

    if !lib_dir.exists() {
        panic!(
            "OPENBLAS_ROOT lib directory does not exist: {}",
            lib_dir.display()
        );
    }
    if !include_dir.exists() {
        panic!(
            "OPENBLAS_ROOT include directory does not exist: {}",
            include_dir.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=openblas");
    println!("cargo:include={}", include_dir.display());
    println!("cargo:rerun-if-env-changed=OPENBLAS_ROOT");
}
