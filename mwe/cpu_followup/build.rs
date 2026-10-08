use std::{env, path::PathBuf};
fn main() {
    if cfg!(feature = "system-mkl") {
        let root = PathBuf::from(env::var("MKLROOT").expect("MKLROOT required"));
        assert!(root.join("include").exists());
        println!(
            "cargo:rustc-link-search=native={}",
            root.join("lib").display()
        );
        println!("cargo:rustc-link-arg=-Wl,--push-state,--no-as-needed");
        for lib in ["mkl_intel_lp64", "mkl_intel_thread", "mkl_core", "iomp5"] {
            println!("cargo:rustc-link-arg=-l{lib}");
        }
        println!("cargo:rustc-link-arg=-Wl,--pop-state");
    } else if cfg!(feature = "system-openblas") {
        let root = PathBuf::from(env::var("OPENBLAS_ROOT").expect("OPENBLAS_ROOT required"));
        println!(
            "cargo:rustc-link-search=native={}",
            root.join("lib").display()
        );
        println!("cargo:rustc-link-arg=-lopenblas");
    }
    println!("cargo:rerun-if-env-changed=MKLROOT");
    println!("cargo:rerun-if-env-changed=OPENBLAS_ROOT");
}
