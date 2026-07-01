use std::env;
use std::path::PathBuf;

fn main() {
    let system_openblas = env::var_os("CARGO_FEATURE_SYSTEM_OPENBLAS").is_some();
    let system_accelerate = env::var_os("CARGO_FEATURE_SYSTEM_ACCELERATE").is_some();
    let system_mkl = env::var_os("CARGO_FEATURE_SYSTEM_MKL").is_some();
    let system_blas_features = [
        ("system-openblas", system_openblas),
        ("system-accelerate", system_accelerate),
        ("system-mkl", system_mkl),
    ];
    let enabled_system_blas_features: Vec<&str> = system_blas_features
        .iter()
        .filter_map(|(name, enabled)| enabled.then_some(*name))
        .collect();
    if enabled_system_blas_features.len() > 1 {
        panic!(
            "features `{}` cannot be enabled together",
            enabled_system_blas_features.join("`, `")
        );
    }

    if system_accelerate {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        return;
    }

    if system_openblas {
        configure_openblas();
    } else if system_mkl {
        configure_mkl();
    }
}

fn configure_openblas() {
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

fn configure_mkl() {
    let root =
        env::var("MKLROOT").expect("MKLROOT must be set when building with feature `system-mkl`");
    let root = PathBuf::from(root);
    let lib_dir = root.join("lib");
    let lib_intel64_dir = root.join("lib").join("intel64");
    let include_dir = root.join("include");

    if !lib_dir.exists() && !lib_intel64_dir.exists() {
        panic!(
            "MKLROOT lib directory does not exist: {}",
            lib_dir.display()
        );
    }
    if !include_dir.exists() {
        panic!(
            "MKLROOT include directory does not exist: {}",
            include_dir.display()
        );
    }

    if lib_dir.exists() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }
    if lib_intel64_dir.exists() {
        println!(
            "cargo:rustc-link-search=native={}",
            lib_intel64_dir.display()
        );
    }
    // Keep the split oneMKL dynamic libraries in DT_NEEDED. With the default
    // Linux --as-needed behavior, mkl_core and mkl_intel_thread can be dropped
    // even though mkl_intel_lp64 resolves symbols from them at runtime.
    println!("cargo:rustc-link-arg=-Wl,--push-state,--no-as-needed");
    println!("cargo:rustc-link-arg=-lmkl_intel_lp64");
    println!("cargo:rustc-link-arg=-lmkl_intel_thread");
    println!("cargo:rustc-link-arg=-lmkl_core");
    println!("cargo:rustc-link-arg=-liomp5");
    println!("cargo:rustc-link-arg=-Wl,--pop-state");
    println!("cargo:include={}", include_dir.display());
    println!("cargo:rerun-if-env-changed=MKLROOT");
}
