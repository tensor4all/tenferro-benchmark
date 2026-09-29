//! Contraction-key streams for `cpu/session_matrix` (tenferro-benchmark #107,
//! tenferro-rs #1946 B4): one timed sample is one whole sequence of `length`
//! public concrete einsum calls (`ij,jk->ik`) in one entered session, so the
//! sequence order controls what any internal plan cache sees:
//!
//! - `fixed`: the same shape every call (best case for a cache);
//! - `mixed`: a bounded round-robin over `keys` shapes (revisits);
//! - `fresh`: a shape never used before in this process for every call
//!   (planning for a new key is inside the timer, by design);
//! - `strides`: one shape whose left operand alternates between `keys`
//!   different leading strides (equal shape, different layout).
//!
//! Inputs for every call of every sample are generated before the timer;
//! outputs are retained until the clock stops and checked afterwards. Plan
//! builds/hits/misses/evictions have no public counter and are reported as
//! "unavailable".

use std::hint::black_box;
use std::time::Instant;

use serde::Deserialize;
use tenferro_einsum::TensorReadEinsumExt;
use tenferro_tensor::{BackendSession, Tensor, TensorRead, TensorView, TypedTensorView};

use super::batch_route::BoxError;

#[derive(Clone, Debug, Deserialize)]
pub struct StreamSpec {
    pub id: String,
    pub stream: String,
    pub length: usize,
    pub keys: usize,
    pub route: serde_json::Value,
}

pub fn load_stream(path: &str, id: &str) -> Result<Option<StreamSpec>, BoxError> {
    let values: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    values
        .into_iter()
        .find(|v| v["id"] == id && v["workload"] == "stream")
        .map(|v| Ok(serde_json::from_value(v)?))
        .transpose()
}

/// One prepared call: storage, view layout and the expected product.
struct Call {
    lhs: Tensor,
    lhs_shape: [usize; 2],
    lhs_strides: [isize; 2],
    rhs: Tensor,
    expected: Vec<f64>,
}

impl Call {
    fn new(m: usize, k: usize, n: usize, lead_pad: usize, seed: usize) -> Result<Self, BoxError> {
        let ld = m + lead_pad;
        let a = |i: usize, p: usize| ((i + 3 * p + seed) % 9) as f64 * 0.125 - 0.5;
        let b = |p: usize, j: usize| ((2 * p + j + seed) % 7) as f64 * 0.25 - 0.75;
        let mut lhs = vec![f64::NAN; ld * k];
        for p in 0..k {
            for i in 0..m {
                lhs[i + ld * p] = a(i, p);
            }
        }
        let rhs: Vec<f64> = (0..k * n).map(|x| b(x % k, x / k)).collect();
        let mut expected = vec![0.0; m * n];
        for j in 0..n {
            for p in 0..k {
                for i in 0..m {
                    expected[i + m * j] += a(i, p) * b(p, j);
                }
            }
        }
        Ok(Self {
            lhs: Tensor::from_vec_col_major(vec![ld * k], lhs)?,
            lhs_shape: [m, k],
            lhs_strides: [1, ld as isize],
            rhs: Tensor::from_vec_col_major(vec![k, n], rhs)?,
            expected,
        })
    }

    fn reads(&self) -> Result<[TensorRead<'_>; 2], BoxError> {
        let view = TypedTensorView::from_slice(
            self.lhs_shape.to_vec(),
            self.lhs_strides.to_vec(),
            0,
            self.lhs.as_slice::<f64>()?,
        )?;
        Ok([
            TensorRead::from_view(TensorView::F64(view)),
            TensorRead::from_tensor(&self.rhs),
        ])
    }
}

/// Every call of every pass, in sequence order.
pub struct Stream {
    passes: Vec<Vec<Call>>,
}

impl Stream {
    pub fn new(spec: &StreamSpec, passes: usize) -> Result<Self, BoxError> {
        let mut fresh = (2..=40usize)
            .flat_map(|m| (2..=12usize).map(move |k| (m, k)))
            .flat_map(|(m, k)| (1..=12usize).map(move |n| (m, k, n)));
        let mixed = [(6, 7, 8), (8, 6, 7), (7, 8, 6), (9, 9, 5)];
        let mut all = Vec::with_capacity(passes);
        for pass in 0..passes {
            let mut calls = Vec::with_capacity(spec.length);
            for i in 0..spec.length {
                let seed = pass * spec.length + i;
                let call = match spec.stream.as_str() {
                    "fixed" => Call::new(8, 8, 8, 0, seed)?,
                    "mixed" => {
                        let (m, k, n) = mixed[i % spec.keys.min(mixed.len())];
                        Call::new(m, k, n, 0, seed)?
                    }
                    "fresh" => {
                        let (m, k, n) = fresh.next().ok_or("fresh key pool exhausted")?;
                        Call::new(m, k, n, 0, seed)?
                    }
                    "strides" => Call::new(8, 8, 8, i % spec.keys, seed)?,
                    other => return Err(format!("unknown stream {other}").into()),
                };
                calls.push(call);
            }
            all.push(calls);
        }
        Ok(Self { passes: all })
    }

    /// Time each pass as one sample. Pass 0 is the explicit untimed
    /// initialization; passes `1..=warmups` are untimed warmups.
    pub fn measure(
        &self,
        session: &mut dyn BackendSession,
        warmups: usize,
    ) -> Result<(Vec<u128>, f64), BoxError> {
        let mut elapsed = Vec::new();
        let mut worst = 0.0_f64;
        for (index, pass) in self.passes.iter().enumerate() {
            let reads = pass
                .iter()
                .map(Call::reads)
                .collect::<Result<Vec<_>, _>>()?;
            let mut outputs = Vec::with_capacity(pass.len());
            let start = Instant::now();
            for read in &reads {
                outputs.push(read.einsum_read("ij,jk->ik", session)?);
            }
            let ns = start.elapsed().as_nanos();
            black_box(&outputs);
            if index > warmups {
                elapsed.push(ns);
            }
            for (output, call) in outputs.iter().zip(pass) {
                for (a, e) in output.as_slice::<f64>()?.iter().zip(&call.expected) {
                    let err = (a - e).abs();
                    if !err.is_finite() || err > 1e-12 * (1.0 + e.abs()) * 16.0 {
                        return Err(format!("stream output {a} != expected {e}").into());
                    }
                    worst = worst.max(err);
                }
            }
        }
        Ok((elapsed, worst))
    }
}
