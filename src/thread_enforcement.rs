//! Startup checks that bind a benchmark process to one requested CPU thread
//! count.
//!
//! A tenferro `CpuBackend` built with `CpuBackend::new()` sizes its worker
//! pool from `RAYON_NUM_THREADS` and falls back to every available core when
//! the variable is unset, so a `--num-threads 1` row could silently run on the
//! whole machine. Benchmarks call [`enforce_thread_request`] before building
//! any backend, construct backends with an explicit thread count, and then
//! confirm the result with [`verify_backend_threads`].

use std::env;

/// Thread-count environment variables that must agree with the request.
///
/// `scripts/thread_env.sh` exports every one of them; a direct invocation
/// with a conflicting value fails instead of mixing thread budgets.
pub const THREAD_COUNT_ENV_VARS: &[&str] = &[
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "VECLIB_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
];

/// Return `VAR=value` for every thread variable set to a different count.
pub fn conflicting_thread_env(
    requested: usize,
    lookup: impl Fn(&str) -> Option<String>,
) -> Vec<String> {
    THREAD_COUNT_ENV_VARS
        .iter()
        .filter_map(|&name| {
            let value = lookup(name)?;
            let trimmed = value.trim();
            if trimmed.is_empty() || trimmed.parse::<usize>().ok() == Some(requested) {
                None
            } else {
                Some(format!("{name}={value}"))
            }
        })
        .collect()
}

/// Validate the thread environment and bound the global Rayon pool.
///
/// Fails when `requested` is zero or when any variable in
/// [`THREAD_COUNT_ENV_VARS`] is set to another value. Unset variables are
/// exported with the requested count before any provider initializes, so BLAS
/// and OpenMP runtimes that read them lazily see the same bound. The global
/// Rayon pool is then built with exactly `requested` workers, which bounds any
/// Rayon work that runs outside a backend-owned pool (tenferro backends with
/// one thread own no pool and fall through to the global one).
pub fn enforce_thread_request(requested: usize) -> Result<(), String> {
    if requested == 0 {
        return Err("requested thread count must be at least 1".into());
    }
    let conflicts = conflicting_thread_env(requested, |name| env::var(name).ok());
    if !conflicts.is_empty() {
        return Err(format!(
            "thread environment conflicts with the requested {requested} threads: {}",
            conflicts.join(", ")
        ));
    }
    for &name in THREAD_COUNT_ENV_VARS {
        if env::var_os(name).map_or(true, |value| value.is_empty()) {
            // Edition 2021: set_var is safe; no other threads exist yet.
            env::set_var(name, requested.to_string());
        }
    }
    // Building fails only if the global pool already exists; the size check
    // below still rejects a pool of the wrong size.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(requested)
        .build_global();
    let global = rayon::current_num_threads();
    if global != requested {
        return Err(format!(
            "global Rayon pool has {global} threads but {requested} were requested"
        ));
    }
    Ok(())
}

/// Fail when a backend's effective thread count differs from the request.
pub fn verify_backend_threads(
    label: &str,
    effective: usize,
    requested: usize,
) -> Result<(), String> {
    if effective == requested {
        Ok(())
    } else {
        Err(format!(
            "{label} runs with {effective} threads but {requested} were requested"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_and_unset_variables_do_not_conflict() {
        let conflicts = conflicting_thread_env(4, |name| match name {
            "RAYON_NUM_THREADS" => Some("4".into()),
            "OMP_NUM_THREADS" => Some(String::new()),
            _ => None,
        });
        assert!(conflicts.is_empty());
    }

    #[test]
    fn different_or_invalid_values_conflict() {
        let conflicts = conflicting_thread_env(1, |name| match name {
            "RAYON_NUM_THREADS" => Some("8".into()),
            "VECLIB_MAXIMUM_THREADS" => Some("all".into()),
            "OMP_NUM_THREADS" => Some("1".into()),
            _ => None,
        });
        assert_eq!(
            conflicts,
            vec!["RAYON_NUM_THREADS=8", "VECLIB_MAXIMUM_THREADS=all"]
        );
    }

    #[test]
    fn backend_thread_mismatch_is_reported() {
        assert!(verify_backend_threads("CpuBackend", 4, 4).is_ok());
        let error = verify_backend_threads("CpuBackend", 10, 1).unwrap_err();
        assert!(error.contains("10 threads"));
    }
}
