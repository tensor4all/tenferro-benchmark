use std::env;
use std::path::PathBuf;

fn main() {
    let system_openblas = env::var_os("CARGO_FEATURE_SYSTEM_OPENBLAS").is_some();
    let system_accelerate = env::var_os("CARGO_FEATURE_SYSTEM_ACCELERATE").is_some();
    if system_openblas && system_accelerate {
        panic!("features `system-openblas` and `system-accelerate` cannot be enabled together");
    }

    if system_accelerate {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        return;
    }

    if !system_openblas {
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
    println!("cargo:rustc-link-arg-bin=publication_gate=-lopenblas");
    println!("cargo:include={}", include_dir.display());
    println!("cargo:rerun-if-env-changed=OPENBLAS_ROOT");
}
