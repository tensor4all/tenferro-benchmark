//! Same-path scouting benchmark: strided-opteinsum (simple recursive executor)
//! vs the current tenferro EagerTensor einsum path (compile/interpret + cache),
//! both driven by the SAME stored contraction path (opt_flops / opt_size).
//!
//! Fixing the path removes the planner-quality confound, so the delta isolates
//! per-step orchestration / intermediate-materialization overhead. Both use the
//! faer GEMM backend, so GEMM cost is shared.
//!
//! Reuses the path->tree construction from tensor4all/strided-rs-benchmark-suite.
//!
//! Run (single-thread parity):
//!   RAYON_NUM_THREADS=1 cargo run --release --features opteinsum-compare \
//!     --bin opteinsum_compare

use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::panic;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;

use strided_opteinsum::{EinsumCode, EinsumNode, EinsumOperand};
use strided_view::StridedArray;

use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::{CpuBackend, CpuBackendKind};
use tenferro_einsum::eager_tensor;
use tenferro_einsum::{ContractionTree, Subscripts};
use tenferro_tensor::Tensor;

#[derive(Deserialize)]
struct Instance {
    name: String,
    format_string_colmajor: String,
    shapes_colmajor: Vec<Vec<usize>>,
    dtype: String,
    num_tensors: usize,
    paths: PathInfo,
}

#[derive(Deserialize)]
struct PathInfo {
    opt_size: PathMeta,
    opt_flops: PathMeta,
}

#[derive(Deserialize)]
struct PathMeta {
    path: Vec<[usize; 2]>,
}

const CASES: &[&str] = &[
    "bin_matmul_256",
    "bin_matmul_1024",
    "bin_batched_matmul_b32_m128_n128_k128",
    "bin_batched_outer_product_compact_j16_k16_o64_t64",
    "str_matrix_chain_multiplication_100",
    "str_mps_varying_inner_product_200",
    "lm_batch_likelihood_sentence_4_4d",
    "str_nw_mera_closed_120",
    "tensornetwork_permutation_light_415",
];

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Remap label chars to a shared alphabetic alphabet that both engines accept.
/// strided-opteinsum requires `char::is_alphabetic()` labels; these instances
/// use symbols like `×`. The same remap is applied to both engines.
fn remap_notation(s: &str) -> String {
    let pool: Vec<char> = (0x41u32..0x30000)
        .filter_map(char::from_u32)
        .filter(|c| c.is_alphabetic())
        .take(8000)
        .collect();
    let mut map = std::collections::HashMap::new();
    let mut next = 0usize;
    let mut out = String::new();
    for c in s.chars() {
        if c == ',' || c == '-' || c == '>' {
            out.push(c);
        } else if !c.is_whitespace() {
            let mapped = *map.entry(c).or_insert_with(|| {
                let m = pool[next];
                next += 1;
                m
            });
            out.push(mapped);
        }
    }
    out
}

/// Select the tenferro CPU GEMM backend via `TENFERRO_CPU_BACKEND_KIND`
/// (default|faer|blas), matching the tenferro-benchmark harness convention.
fn cpu_backend() -> Result<CpuBackend, String> {
    match std::env::var("TENFERRO_CPU_BACKEND_KIND")
        .unwrap_or_else(|_| "default".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "default" | "" => Ok(CpuBackend::new()),
        "faer" => CpuBackend::with_kind(CpuBackendKind::Faer).map_err(|e| e.to_string()),
        "blas" => CpuBackend::with_kind(CpuBackendKind::Blas).map_err(|e| e.to_string()),
        other => Err(format!("unknown TENFERRO_CPU_BACKEND_KIND={other}")),
    }
}

fn load(name: &str) -> Option<Instance> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/instances")
        .join(format!("{name}.json"));
    let json = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&json).ok()
}

// ---------------------------------------------------------------------------
// strided-opteinsum, fixed path (reused from strided-rs-benchmark-suite)
// ---------------------------------------------------------------------------

fn parse_format_string(s: &str) -> (Vec<Vec<char>>, Vec<char>) {
    let (inputs_str, output_str) = s.split_once("->").expect("format_string must contain '->'");
    let input_indices = inputs_str
        .split(',')
        .map(|operand| operand.chars().collect())
        .collect();
    let output_indices = output_str.chars().collect();
    (input_indices, output_indices)
}

fn build_contraction_tree(input_indices: &[Vec<char>], path: &[[usize; 2]]) -> EinsumNode {
    let mut nodes: Vec<EinsumNode> = input_indices
        .iter()
        .enumerate()
        .map(|(i, ids)| EinsumNode::Leaf {
            ids: ids.clone(),
            tensor_index: i,
        })
        .collect();
    for &pair in path {
        let (i, j) = if pair[0] < pair[1] {
            (pair[0], pair[1])
        } else {
            (pair[1], pair[0])
        };
        let node_j = nodes.remove(j);
        let node_i = nodes.remove(i);
        nodes.push(EinsumNode::Contract {
            args: vec![node_i, node_j],
        });
    }
    nodes.pop().expect("path reduces to a single node")
}

fn time_opteinsum(
    inst: &Instance,
    path: &[[usize; 2]],
    warmups: usize,
    runs: usize,
) -> Result<Duration, String> {
    if inst.dtype != "float64" {
        return Err(format!("dtype {} unsupported", inst.dtype));
    }
    let notation = remap_notation(&inst.format_string_colmajor);
    let (input_indices, output_ids) = parse_format_string(&notation);
    let root = build_contraction_tree(&input_indices, path);
    let code = EinsumCode { root, output_ids };

    let make_operands = || -> Vec<EinsumOperand<'static>> {
        inst.shapes_colmajor
            .iter()
            .map(|shape| EinsumOperand::from(StridedArray::<f64>::col_major(shape)))
            .collect()
    };

    for _ in 0..warmups {
        let operands = make_operands();
        let r = panic::catch_unwind(panic::AssertUnwindSafe(|| code.evaluate(operands, None)))
            .map_err(|_| "panic in opteinsum".to_string())?
            .map_err(|e| format!("{e}"))?;
        black_box(r.dims().len());
    }
    let mut d = Vec::with_capacity(runs);
    for _ in 0..runs {
        let operands = make_operands();
        let t0 = Instant::now();
        let r = panic::catch_unwind(panic::AssertUnwindSafe(|| code.evaluate(operands, None)))
            .map_err(|_| "panic in opteinsum".to_string())?
            .map_err(|e| format!("{e}"))?;
        d.push(t0.elapsed());
        black_box(r.dims().len());
    }
    Ok(median(d))
}

// ---------------------------------------------------------------------------
// tenferro EagerTensor, same fixed path (pairwise binary einsum walk)
// ---------------------------------------------------------------------------

fn append_unique(labels: &mut Vec<u32>, label: u32) {
    if !labels.contains(&label) {
        labels.push(label);
    }
}

/// Intermediate output labels for a binary step (copied from the eager harness).
fn intermediate_output_labels(
    subscripts: &[Vec<u32>],
    final_output: &[u32],
    lhs: usize,
    rhs: usize,
) -> Vec<u32> {
    let mut outside: HashMap<u32, usize> = HashMap::new();
    for (i, labels) in subscripts.iter().enumerate() {
        if i == lhs || i == rhs {
            continue;
        }
        for &l in labels {
            *outside.entry(l).or_default() += 1;
        }
    }
    let mut in_order = Vec::new();
    for &l in &subscripts[lhs] {
        append_unique(&mut in_order, l);
    }
    for &l in &subscripts[rhs] {
        append_unique(&mut in_order, l);
    }
    let set: HashSet<u32> = in_order.iter().copied().collect();
    let mut out = Vec::new();
    for &l in final_output {
        if set.contains(&l) {
            append_unique(&mut out, l);
        }
    }
    for &l in &in_order {
        if outside.get(&l).copied().unwrap_or(0) > 0 {
            append_unique(&mut out, l);
        }
    }
    out
}

fn contract_once_tenferro(
    operands: &[EagerTensor],
    inputs: &[Vec<u32>],
    final_output: &[u32],
    path: &[[usize; 2]],
    materialize: bool,
) -> Result<(), String> {
    let mut subs = inputs.to_vec();
    let mut ops = operands.to_vec();
    for &pair in path {
        let (i, j) = if pair[0] < pair[1] {
            (pair[0], pair[1])
        } else {
            (pair[1], pair[0])
        };
        let out_labels = intermediate_output_labels(&subs, final_output, i, j);
        let binary = Subscripts::new(&[&subs[i], &subs[j]], &out_labels);
        let rhs = ops.remove(j);
        let lhs = ops.remove(i);
        let refs = [&lhs, &rhs];
        let binary_subs = (&binary).into();
        let res = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            eager_tensor::einsum_subscripts(&refs, &binary_subs)
        }))
        .map_err(|_| "panic in tenferro".to_string())?
        .map_err(|e| format!("{e}"))?;
        ops.push(res);
        subs.remove(j);
        subs.remove(i);
        subs.push(out_labels);
    }
    let out = ops.pop().ok_or("no output")?;
    if materialize {
        black_box(out.data().shape().len());
    } else {
        black_box(&out.tensor_read());
    }
    Ok(())
}

fn time_tenferro(
    inst: &Instance,
    path: &[[usize; 2]],
    warmups: usize,
    runs: usize,
    materialize: bool,
) -> Result<Duration, String> {
    if inst.dtype != "float64" {
        return Err(format!("dtype {} unsupported", inst.dtype));
    }
    let notation = remap_notation(&inst.format_string_colmajor);
    let parsed = Subscripts::parse(&notation).map_err(|e| format!("{e}"))?;
    let inputs = parsed.inputs.clone();
    let final_output = parsed.output.clone();

    let ctx: Arc<EagerRuntime> = EagerRuntime::with_cpu_backend(cpu_backend()?);
    let operands: Vec<EagerTensor> = inst
        .shapes_colmajor
        .iter()
        .map(|shape| {
            let tensor = Tensor::F64(tenferro_tensor::TypedTensor::<f64>::zeros(shape.clone()));
            EagerTensor::from_tensor_in(tensor, Arc::clone(&ctx))
        })
        .collect();

    for _ in 0..warmups {
        contract_once_tenferro(&operands, &inputs, &final_output, path, materialize)?;
    }
    let mut d = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        contract_once_tenferro(&operands, &inputs, &final_output, path, materialize)?;
        d.push(t0.elapsed());
    }
    Ok(median(d))
}

/// Convert an opt_einsum/cotengra path to `ContractionTree::from_pairs` pairs.
fn path_to_pairs(n_inputs: usize, path: &[[usize; 2]]) -> Vec<(usize, usize)> {
    let mut available: Vec<usize> = (0..n_inputs).collect();
    let mut pairs = Vec::with_capacity(path.len());
    for (step, &pair) in path.iter().enumerate() {
        let (i, j) = if pair[0] < pair[1] {
            (pair[0], pair[1])
        } else {
            (pair[1], pair[0])
        };
        pairs.push((available[i], available[j]));
        available.remove(j);
        available.remove(i);
        available.push(n_inputs + step);
    }
    pairs
}

/// Time the prototype whole-program eager executor on the SAME `opt_flops` path:
/// ONE call to `einsum_whole_program_untracked`, so the whole contraction runs
/// in a single backend session instead of per-step eager ops.
fn time_tenferro_whole(
    inst: &Instance,
    path: &[[usize; 2]],
    warmups: usize,
    runs: usize,
    materialize: bool,
) -> Result<Duration, String> {
    if inst.dtype != "float64" {
        return Err(format!("dtype {} unsupported", inst.dtype));
    }
    let notation = remap_notation(&inst.format_string_colmajor);
    let subs = Subscripts::parse(&notation).map_err(|e| format!("{e}"))?;
    let shape_refs: Vec<&[usize]> = inst.shapes_colmajor.iter().map(|s| s.as_slice()).collect();
    let pairs = path_to_pairs(inst.num_tensors, path);
    let tree = ContractionTree::from_pairs(&subs, &shape_refs, &pairs).map_err(|e| format!("{e}"))?;

    let ctx: Arc<EagerRuntime> = EagerRuntime::with_cpu_backend(cpu_backend()?);
    let operands: Vec<EagerTensor> = inst
        .shapes_colmajor
        .iter()
        .map(|shape| {
            let tensor = Tensor::F64(tenferro_tensor::TypedTensor::<f64>::zeros(shape.clone()));
            EagerTensor::from_tensor_in(tensor, Arc::clone(&ctx))
        })
        .collect();

    let call = || -> Result<(), String> {
        let refs: Vec<&EagerTensor> = operands.iter().collect();
        let out = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            eager_tensor::einsum_whole_program_untracked(&refs, &tree)
        }))
        .map_err(|_| "panic in tenferro whole-program".to_string())?
        .map_err(|e| format!("{e}"))?;
        if materialize {
            black_box(out.data().shape().len());
        } else {
            black_box(&out.tensor_read());
        }
        Ok(())
    };

    for _ in 0..warmups {
        call()?;
    }
    let mut d = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        call()?;
        d.push(t0.elapsed());
    }
    Ok(median(d))
}

fn main() {
    let warmups = env_usize("BENCH_WARMUPS", 3);
    let runs = env_usize("BENCH_RUNS", 15);
    let strategy = std::env::var("BENCH_STRATEGY").unwrap_or_else(|_| "opt_flops".into());

    println!(
        "# same-path: strided-opteinsum(faer) vs tenferro-eager (strategy={strategy}, warmups={warmups}, runs={runs}, TENFERRO_CPU_BACKEND_KIND={}, RAYON_NUM_THREADS={}, VECLIB_MAXIMUM_THREADS={})",
        std::env::var("TENFERRO_CPU_BACKEND_KIND").unwrap_or_else(|_| "default".into()),
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "<unset>".into()),
        std::env::var("VECLIB_MAXIMUM_THREADS").unwrap_or_else(|_| "<unset>".into()),
    );
    println!(
        "{:<40} {:>6} {:>10} {:>11} {:>11} {:>11} {:>11}",
        "case", "ntens", "opteinsum", "tf-pair(rd)", "tf-pair(dt)", "tf-whole(rd)", "tf-whole(dt)"
    );

    let cases: Vec<String> = match std::env::var("BENCH_CASES") {
        Ok(v) if !v.trim().is_empty() => v.split(',').map(|s| s.trim().to_string()).collect(),
        _ => CASES.iter().map(|s| s.to_string()).collect(),
    };

    for name in &cases {
        let Some(inst) = load(name) else {
            println!("{name:<46} (instance not found)");
            continue;
        };
        let path = match strategy.as_str() {
            "opt_size" => &inst.paths.opt_size.path,
            _ => &inst.paths.opt_flops.path,
        }
        .clone();

        eprintln!("[running] {name} (n={}) ...", inst.num_tensors);
        let opt = time_opteinsum(&inst, &path, warmups, runs);
        let tf_read = time_tenferro(&inst, &path, warmups, runs, false);
        let tf_data = time_tenferro(&inst, &path, warmups, runs, true);
        let tw_read = time_tenferro_whole(&inst, &path, warmups, runs, false);
        let tw_data = time_tenferro_whole(&inst, &path, warmups, runs, true);
        eprintln!("[running] {name} done");

        let cell = |r: &Result<Duration, String>| match r {
            Ok(d) => format!("{:.4}ms", ms(*d)),
            Err(_) => "ERR".to_string(),
        };
        println!(
            "{:<40} {:>6} {:>10} {:>11} {:>11} {:>11} {:>11}",
            inst.name,
            inst.num_tensors,
            cell(&opt),
            cell(&tf_read),
            cell(&tf_data),
            cell(&tw_read),
            cell(&tw_data),
        );
        for (label, r) in [
            ("opteinsum", &opt),
            ("tf-pair-read", &tf_read),
            ("tf-pair-data", &tf_data),
            ("tf-whole-read", &tw_read),
            ("tf-whole-data", &tw_data),
        ] {
            if let Err(e) = r {
                eprintln!("  {name} {label} ERR: {e}");
            }
        }
    }
}
