//! CPU provider selection for the benchmark backends.
//!
//! A default build measures tenferro's compiled backend unchanged. With the
//! `tprims` Cargo feature, every CPU backend built through [`configure`]
//! installs the tprims GEMM and general-contraction providers from
//! `extern/tenferro-rs/ext/tenferro-cpu-tprims` (tensor4all/tprims-rs phase
//! 1e): the acceptance comparison is a default build against a
//! `--features tprims` build of the same commit.
//!
//! With the feature and `TPRIMS_SHAPE_LOG=<file>`, every provider call is
//! also appended to `<file>` as one JSON line (operation, dtype, operand
//! shapes and strides, contraction axes, elapsed nanoseconds, outcome); the
//! lines feed `scripts/tprims_shape_corpus.py`, which builds the corpus for
//! tprims-rs's `contract --corpus` / `blas --corpus`.

use tenferro_cpu::CpuBackend;

/// Name of the CPU provider configuration this binary measures, for run
/// metadata: `default` or `tprims`.
pub fn describe() -> &'static str {
    if cfg!(feature = "tprims") {
        "tprims"
    } else {
        "default"
    }
}

/// Install the tprims providers on `backend` when built with the `tprims`
/// feature; otherwise return it unchanged. The bundle keeps the backend's
/// own kind as the fallback for everything tprims does not handle.
///
/// # Errors
///
/// The provider bundle cannot be built or installed on this backend.
pub fn configure(backend: CpuBackend) -> Result<CpuBackend, String> {
    #[cfg(feature = "tprims")]
    {
        tprims::install(backend)
    }
    #[cfg(not(feature = "tprims"))]
    {
        Ok(backend)
    }
}

#[cfg(feature = "tprims")]
mod tprims {
    use std::fs::{File, OpenOptions};
    use std::io::{BufWriter, Write};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::Instant;

    use serde_json::json;
    use tenferro_cpu::provider::{
        CpuBatchedMatrixLayout, CpuDotGeneralRequest, CpuExecutionContext, CpuGemmProvider,
        CpuGemmRequest, CpuGeneralContractionProvider, CpuGroupedGemmRequest, CpuProviderOutcome,
    };
    use tenferro_cpu::{CpuBackend, CpuProviderBundle, CpuProviderExecutionCapabilities};
    use tenferro_cpu_tprims::TprimsProvider;
    use tenferro_tensor::{col_major_strides, TensorRead, TensorView};

    pub(super) fn install(backend: CpuBackend) -> Result<CpuBackend, String> {
        let bundle = CpuProviderBundle::builder(backend.kind())
            .gemm_provider(Arc::new(Logged))
            .prefer_general_contraction_provider(Arc::new(Logged))
            .build()
            .map_err(|e| e.to_string())?;
        backend
            .with_provider_bundle(bundle)
            .map_err(|e| e.to_string())
    }

    /// The shape log named by `TPRIMS_SHAPE_LOG`, opened once for appending.
    fn log() -> Option<&'static Mutex<BufWriter<File>>> {
        static LOG: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();
        LOG.get_or_init(|| {
            let path = std::env::var_os("TPRIMS_SHAPE_LOG")?;
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .ok()?;
            Some(Mutex::new(BufWriter::new(file)))
        })
        .as_ref()
    }

    fn strides(r: &TensorRead<'_>) -> Vec<isize> {
        let view = match r {
            TensorRead::Tensor(t) => {
                return col_major_strides(t.shape()).unwrap_or_default();
            }
            TensorRead::View(v) => v,
        };
        match view {
            TensorView::F32(v) => v.strides().to_vec(),
            TensorView::F64(v) => v.strides().to_vec(),
            TensorView::C32(v) => v.strides().to_vec(),
            TensorView::C64(v) => v.strides().to_vec(),
            other => col_major_strides(other.shape()).unwrap_or_default(),
        }
    }

    fn layout(l: CpuBatchedMatrixLayout) -> serde_json::Value {
        json!([l.row_stride(), l.column_stride(), l.batch_stride()])
    }

    fn record(
        entry: serde_json::Value,
        start: Instant,
        outcome: &tenferro_tensor::Result<CpuProviderOutcome>,
    ) {
        let Some(log) = log() else { return };
        let mut entry = entry;
        entry["ns"] = json!(start.elapsed().as_nanos() as u64);
        entry["outcome"] = json!(match outcome {
            Ok(CpuProviderOutcome::Executed) => "executed",
            Ok(CpuProviderOutcome::Unsupported(_)) => "unsupported",
            Err(_) => "error",
        });
        if let Ok(mut w) = log.lock() {
            let _ = writeln!(w, "{entry}");
            let _ = w.flush();
        }
    }

    fn gemm_entry(op: &str, r: &CpuGemmRequest<'_, '_, '_>) -> serde_json::Value {
        json!({
            "op": op,
            "dtype": format!("{:?}", r.lhs().dtype()),
            "m": r.rows(), "n": r.columns(), "k": r.contracted(), "batch": r.batch_count(),
            "a": layout(r.lhs_layout()), "b": layout(r.rhs_layout()), "c": layout(r.output_layout()),
            "conj": [r.accumulation().lhs_conj, r.accumulation().rhs_conj],
        })
    }

    /// `TprimsProvider` plus the optional shape log.
    #[derive(Debug)]
    struct Logged;

    impl CpuGemmProvider for Logged {
        fn execution_capabilities(&self) -> CpuProviderExecutionCapabilities {
            CpuGemmProvider::execution_capabilities(&TprimsProvider::new())
        }
        fn gemm(
            &self,
            c: &CpuExecutionContext<'_>,
            r: CpuGemmRequest<'_, '_, '_>,
        ) -> tenferro_tensor::Result<CpuProviderOutcome> {
            let entry = log().map(|_| gemm_entry("gemm", &r));
            let start = Instant::now();
            let out = TprimsProvider::new().gemm(c, r);
            if let Some(e) = entry {
                record(e, start, &out);
            }
            out
        }
        fn strided_batched_gemm(
            &self,
            c: &CpuExecutionContext<'_>,
            r: CpuGemmRequest<'_, '_, '_>,
        ) -> tenferro_tensor::Result<CpuProviderOutcome> {
            let entry = log().map(|_| gemm_entry("gemm_batched", &r));
            let start = Instant::now();
            let out = TprimsProvider::new().strided_batched_gemm(c, r);
            if let Some(e) = entry {
                record(e, start, &out);
            }
            out
        }
        fn grouped_gemm(
            &self,
            c: &CpuExecutionContext<'_>,
            r: CpuGroupedGemmRequest<'_, '_, '_>,
        ) -> tenferro_tensor::Result<CpuProviderOutcome> {
            let entry = log().map(|_| {
                let jobs: Vec<_> =
                    r.jobs().iter().map(|j| [j.rows(), j.contracted(), j.cols()]).collect();
                json!({"op": "gemm_grouped", "dtype": format!("{:?}", r.lhs().dtype()), "jobs": jobs})
            });
            let start = Instant::now();
            let out = TprimsProvider::new().grouped_gemm(c, r);
            if let Some(e) = entry {
                record(e, start, &out);
            }
            out
        }
    }

    impl CpuGeneralContractionProvider for Logged {
        fn execution_capabilities(&self) -> CpuProviderExecutionCapabilities {
            CpuGeneralContractionProvider::execution_capabilities(&TprimsProvider::new())
        }
        fn dot_general(
            &self,
            c: &CpuExecutionContext<'_>,
            r: CpuDotGeneralRequest<'_, '_, '_>,
        ) -> tenferro_tensor::Result<CpuProviderOutcome> {
            let entry = log().map(|_| {
                let axes = r.axes();
                let (lc, rc): (Vec<usize>, Vec<usize>) = axes.contracting_pairs().unzip();
                let (lb, rb): (Vec<usize>, Vec<usize>) = axes.batch_pairs().unzip();
                json!({
                    "op": "dot_general",
                    "dtype": format!("{:?}", r.lhs().dtype()),
                    "a": {"dims": r.lhs().shape(), "strides": strides(r.lhs())},
                    "b": {"dims": r.rhs().shape(), "strides": strides(r.rhs())},
                    "lc": lc, "rc": rc, "lb": lb, "rb": rb,
                    "conj": [r.accumulation().lhs_conj, r.accumulation().rhs_conj],
                })
            });
            let start = Instant::now();
            let out = TprimsProvider::new().dot_general(c, r);
            if let Some(e) = entry {
                record(e, start, &out);
            }
            out
        }
    }
}
