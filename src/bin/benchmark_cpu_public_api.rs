//! CPU public API benchmark runner for tenferro-rs APIs not covered by the
//! focused FFT/einsum/permutation suites.

use std::env;
use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use num_complex::Complex64;
use tenferro_cpu::{CpuBackend, CpuBackendKind};
use tenferro_linalg::TensorLinalgExt;
use tenferro_tensor::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig, Tensor,
    TensorAnalytic, TensorDot, TensorElementwise, TensorIndexing, TensorReduction,
};

type BenchResult<T> = Result<T, Box<dyn std::error::Error>>;

struct Args {
    output: PathBuf,
    num_threads: usize,
    runs: usize,
    warmups: usize,
}

struct Case {
    suite: &'static str,
    benchmark: &'static str,
    dtype: &'static str,
    shape: &'static str,
    notes: &'static str,
    run: fn(&mut CpuBackend) -> tenferro_tensor::Result<()>,
}

fn main() -> BenchResult<()> {
    let args = parse_args()?;
    let append = args.output.exists();
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.output)?;
    let mut writer = BufWriter::new(file);
    if !append {
        writeln!(
            writer,
            "suite,benchmark,dtype,threads,shape,backend,median_ms,iqr_ms,status,notes"
        )?;
    }

    let mut backend = cpu_backend_from_env()?;
    for case in cases() {
        emit_case(&mut writer, &args, &mut backend, case)?;
    }
    writer.flush()?;
    Ok(())
}

fn parse_args() -> BenchResult<Args> {
    let mut output = None;
    let mut num_threads = None;
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--output" => output = iter.next().map(PathBuf::from),
            "--num-threads" => {
                num_threads = Some(
                    iter.next()
                        .ok_or("--num-threads requires a value")?
                        .parse::<usize>()?,
                );
            }
            "-h" | "--help" => {
                println!("Usage: benchmark_cpu_public_api --output <csv> [--num-threads N]");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    let profile = env::var("PUBLICATION_GATE_PROFILE")
        .unwrap_or_else(|_| "quick".to_string())
        .to_ascii_lowercase();
    Ok(Args {
        output: output.ok_or("--output is required")?,
        num_threads: num_threads
            .or_else(|| env::var("RAYON_NUM_THREADS").ok()?.parse::<usize>().ok())
            .unwrap_or(1),
        runs: env::var("BENCH_RUNS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(if profile == "full" { 15 } else { 7 }),
        warmups: env::var("BENCH_WARMUPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3),
    })
}

fn cases() -> Vec<Case> {
    vec![
        // Elementwise and reductions (#73).
        elem("add", "f64", "262144", "binary elementwise", add_f64),
        elem("sub", "f64", "262144", "binary elementwise", sub_f64),
        elem("mul", "f64", "262144", "binary elementwise", mul_f64),
        elem("div", "f64", "262144", "binary elementwise", div_f64),
        elem("rem", "f64", "262144", "binary elementwise", rem_f64),
        elem("neg", "f64", "262144", "unary elementwise", neg_f64),
        elem("abs", "f64", "262144", "unary elementwise", abs_f64),
        elem("sign", "f64", "262144", "unary elementwise", sign_f64),
        elem(
            "maximum",
            "f64",
            "262144",
            "binary elementwise",
            maximum_f64,
        ),
        elem(
            "minimum",
            "f64",
            "262144",
            "binary elementwise",
            minimum_f64,
        ),
        elem(
            "compare_lt",
            "f64",
            "262144",
            "ordered compare",
            compare_lt_f64,
        ),
        elem("select", "f64", "262144", "ternary select", select_f64),
        elem(
            "clamp",
            "f64",
            "262144",
            "clamp with tensor bounds",
            clamp_f64,
        ),
        elem("exp", "f64", "262144", "analytic unary", exp_f64),
        elem("log", "f64", "262144", "analytic unary", log_f64),
        elem("sin", "f64", "262144", "analytic unary", sin_f64),
        elem("cos", "f64", "262144", "analytic unary", cos_f64),
        elem("tanh", "f64", "262144", "analytic unary", tanh_f64),
        elem("sqrt", "f64", "262144", "analytic unary", sqrt_f64),
        elem("rsqrt", "f64", "262144", "analytic unary", rsqrt_f64),
        elem("pow", "f64", "262144", "binary analytic", pow_f64),
        elem("expm1", "f64", "262144", "analytic unary", expm1_f64),
        elem("log1p", "f64", "262144", "analytic unary", log1p_f64),
        elem(
            "chain_log1p_exp_mul",
            "f64",
            "262144",
            "short elementwise chain",
            chain_f64,
        ),
        elem(
            "reduce_sum_all",
            "f64",
            "256x1024",
            "full reduction",
            reduce_sum_all_f64,
        ),
        elem(
            "reduce_prod_all",
            "f64",
            "256x1024",
            "full reduction",
            reduce_prod_all_f64,
        ),
        elem(
            "reduce_max_axis0",
            "f64",
            "256x1024",
            "axis reduction",
            reduce_max_axis0_f64,
        ),
        elem(
            "reduce_min_axis1",
            "f64",
            "256x1024",
            "axis reduction",
            reduce_min_axis1_f64,
        ),
        // Indexing/layout (#72).
        idx(
            "gather",
            "f64",
            "65536",
            "1D StableHLO-style gather",
            gather_f64,
        ),
        idx(
            "scatter",
            "f64",
            "65536",
            "1D StableHLO-style scatter",
            scatter_f64,
        ),
        idx("slice", "f64", "65536", "static slice", slice_f64),
        idx(
            "dynamic_slice",
            "f64",
            "65536",
            "runtime-start slice",
            dynamic_slice_f64,
        ),
        idx(
            "dynamic_update_slice",
            "f64",
            "65536",
            "runtime-start update",
            dynamic_update_slice_f64,
        ),
        idx("pad", "f64", "65536", "edge padding", pad_f64),
        idx(
            "concatenate",
            "f64",
            "32768+32768",
            "concatenate along axis 0",
            concatenate_f64,
        ),
        idx("reverse", "f64", "65536", "reverse axis 0", reverse_f64),
        // Uncovered linalg (#71).
        lin("cholesky", "f64", "128x128", "SPD input", cholesky_f64),
        lin("eig", "f64", "64x64", "general input", eig_f64),
        lin(
            "eigvals",
            "f64",
            "64x64",
            "general input values only",
            eigvals_f64,
        ),
        lin(
            "eigvalsh",
            "f64",
            "128x128",
            "SPD input values only",
            eigvalsh_f64,
        ),
        lin(
            "triangular_solve",
            "f64",
            "128x128,rhs=16",
            "lower-triangular solve",
            triangular_solve_f64,
        ),
        lin("det", "f64", "128x128", "well-conditioned input", det_f64),
        lin(
            "slogdet",
            "f64",
            "128x128",
            "well-conditioned input",
            slogdet_f64,
        ),
        lin("inv", "f64", "128x128", "well-conditioned input", inv_f64),
        lin("pinv", "f64", "128x64", "rectangular input", pinv_f64),
        lin("norm_fro", "f64", "256x256", "Frobenius norm", norm_f64),
        lin(
            "full_piv_lu_solve",
            "f64",
            "64x64,rhs=8",
            "tenferro full pivot solve; PyTorch uses direct solve",
            full_piv_lu_solve_f64,
        ),
        // Complex coverage (#74).
        cplx("conj", "c64", "65536", "complex elementwise", conj_c64),
        cplx("mul", "c64", "65536", "complex elementwise", mul_c64),
        cplx("div", "c64", "65536", "complex elementwise", div_c64),
        cplx("exp", "c64", "65536", "complex analytic", exp_c64),
        cplx("log", "c64", "65536", "complex analytic", log_c64),
        cplx(
            "dot_general_conj",
            "c64",
            "128x128",
            "complex matrix multiply",
            dot_c64,
        ),
        cplx("svd", "c64", "32x32", "complex SVD", svd_c64),
        cplx("qr", "c64", "32x32", "complex QR", qr_c64),
        cplx("eig", "c64", "32x32", "complex eig", eig_c64),
        cplx("solve", "c64", "32x32,rhs=4", "complex solve", solve_c64),
        cplx(
            "cholesky",
            "c64",
            "32x32",
            "Hermitian positive definite",
            cholesky_c64,
        ),
        cplx(
            "norm_fro",
            "c64",
            "64x64",
            "complex Frobenius norm",
            norm_c64,
        ),
    ]
}

fn elem(
    benchmark: &'static str,
    dtype: &'static str,
    shape: &'static str,
    notes: &'static str,
    run: fn(&mut CpuBackend) -> tenferro_tensor::Result<()>,
) -> Case {
    Case {
        suite: "cpu/elementwise_reduction",
        benchmark,
        dtype,
        shape,
        notes,
        run,
    }
}

fn idx(
    benchmark: &'static str,
    dtype: &'static str,
    shape: &'static str,
    notes: &'static str,
    run: fn(&mut CpuBackend) -> tenferro_tensor::Result<()>,
) -> Case {
    Case {
        suite: "cpu/indexing_layout",
        benchmark,
        dtype,
        shape,
        notes,
        run,
    }
}

fn lin(
    benchmark: &'static str,
    dtype: &'static str,
    shape: &'static str,
    notes: &'static str,
    run: fn(&mut CpuBackend) -> tenferro_tensor::Result<()>,
) -> Case {
    Case {
        suite: "cpu/linalg_uncovered",
        benchmark,
        dtype,
        shape,
        notes,
        run,
    }
}

fn cplx(
    benchmark: &'static str,
    dtype: &'static str,
    shape: &'static str,
    notes: &'static str,
    run: fn(&mut CpuBackend) -> tenferro_tensor::Result<()>,
) -> Case {
    Case {
        suite: "cpu/complex",
        benchmark,
        dtype,
        shape,
        notes,
        run,
    }
}

fn emit_case(
    writer: &mut impl Write,
    args: &Args,
    backend: &mut CpuBackend,
    case: Case,
) -> BenchResult<()> {
    match time_case(args, backend, case.run) {
        Ok((median_ms, iqr_ms)) => {
            writeln!(
                writer,
                "{},{},{},{},\"{}\",tenferro-eager,{median_ms:.6},{iqr_ms:.6},ok,\"{}\"",
                case.suite,
                case.benchmark,
                case.dtype,
                args.num_threads,
                csv_escape(case.shape),
                csv_escape(case.notes),
            )?;
        }
        Err(err) => {
            writeln!(
                writer,
                "{},{},{},{},\"{}\",tenferro-eager,,,failed,\"{}\"",
                case.suite,
                case.benchmark,
                case.dtype,
                args.num_threads,
                csv_escape(case.shape),
                csv_escape(&err.to_string()),
            )?;
        }
    }
    Ok(())
}

fn time_case(
    args: &Args,
    backend: &mut CpuBackend,
    run: fn(&mut CpuBackend) -> tenferro_tensor::Result<()>,
) -> BenchResult<(f64, f64)> {
    for _ in 0..args.warmups {
        run(backend)?;
    }
    let mut times = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let start = Instant::now();
        run(backend)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(median_iqr(&times))
}

fn median_iqr(times: &[f64]) -> (f64, f64) {
    let mut values = times.to_vec();
    values.sort_by(|a, b| a.total_cmp(b));
    let median = if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    } else {
        values[values.len() / 2]
    };
    let iqr = values[(3 * values.len()) / 4] - values[values.len() / 4];
    (median, iqr)
}

fn cpu_backend_from_env() -> BenchResult<CpuBackend> {
    match env::var("TENFERRO_CPU_BACKEND_KIND")
        .unwrap_or_else(|_| "default".to_string())
        .as_str()
    {
        "" | "default" => Ok(CpuBackend::new()),
        "blas" => Ok(CpuBackend::with_kind(CpuBackendKind::Blas)?),
        "faer" => Ok(CpuBackend::with_kind(CpuBackendKind::Faer)?),
        other => Err(format!("unsupported TENFERRO_CPU_BACKEND_KIND={other}").into()),
    }
}

fn csv_escape(value: &str) -> String {
    value.replace('"', "\"\"")
}

fn consume(tensor: Tensor) {
    black_box(tensor.shape().len());
    black_box(tensor.dtype());
}

fn consume_many(values: impl IntoIterator<Item = Tensor>) {
    for value in values {
        consume(value);
    }
}

fn data_f64(n: usize, seed: u64) -> Vec<f64> {
    (0..n).map(|i| pseudo_value(i, seed)).collect()
}

fn positive_data_f64(n: usize, seed: u64) -> Vec<f64> {
    (0..n).map(|i| 0.25 + pseudo_value(i, seed).abs()).collect()
}

fn data_i64(n: usize, modulus: usize) -> Vec<i64> {
    (0..n).map(|i| ((i * 37 + 11) % modulus) as i64).collect()
}

fn data_bool(n: usize) -> Vec<bool> {
    (0..n).map(|i| i % 3 == 0).collect()
}

fn data_c64(n: usize, seed: u64) -> Vec<Complex64> {
    (0..n)
        .map(|i| Complex64::new(pseudo_value(i, seed), pseudo_value(i, seed + 1)))
        .collect()
}

fn pseudo_value(i: usize, seed: u64) -> f64 {
    let x = (i as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(seed.wrapping_mul(1_442_695_040_888_963_407));
    ((x % 2048) as f64 - 1024.0) / 1024.0
}

fn tensor_f64(shape: &[usize], seed: u64) -> Tensor {
    Tensor::from_vec_col_major(shape.to_vec(), data_f64(shape.iter().product(), seed)).unwrap()
}

fn tensor_f64_positive(shape: &[usize], seed: u64) -> Tensor {
    Tensor::from_vec_col_major(
        shape.to_vec(),
        positive_data_f64(shape.iter().product(), seed),
    )
    .unwrap()
}

fn tensor_c64(shape: &[usize], seed: u64) -> Tensor {
    Tensor::from_vec_col_major(shape.to_vec(), data_c64(shape.iter().product(), seed)).unwrap()
}

fn well_conditioned(n: usize, seed: u64) -> Tensor {
    let mut values = data_f64(n * n, seed);
    for j in 0..n {
        values[j + j * n] += 2.0 + j as f64 / n as f64;
    }
    Tensor::from_vec_col_major(vec![n, n], values).unwrap()
}

fn lower_triangular(n: usize, seed: u64) -> Tensor {
    let mut values = vec![0.0; n * n];
    for col in 0..n {
        for row in col..n {
            values[row + col * n] = if row == col {
                2.0 + row as f64 / n as f64
            } else {
                0.05 * pseudo_value(row + col * n, seed)
            };
        }
    }
    Tensor::from_vec_col_major(vec![n, n], values).unwrap()
}

fn spd(n: usize, seed: u64) -> Tensor {
    let mut values = vec![0.0; n * n];
    for col in 0..n {
        values[col + col * n] = 2.0 + col as f64 / n as f64;
        for row in (col + 1)..n {
            let value = 0.01 * pseudo_value(row + col * n, seed);
            values[row + col * n] = value;
            values[col + row * n] = value;
        }
    }
    Tensor::from_vec_col_major(vec![n, n], values).unwrap()
}

fn hpd_c64(n: usize, seed: u64) -> Tensor {
    let mut values = vec![Complex64::new(0.0, 0.0); n * n];
    for col in 0..n {
        for row in 0..n {
            values[row + col * n] = if row == col {
                Complex64::new(2.0 + row as f64 / n as f64, 0.0)
            } else if row > col {
                Complex64::new(
                    0.01 * pseudo_value(row + col * n, seed),
                    0.01 * pseudo_value(row + col * n, seed + 1),
                )
            } else {
                values[col + row * n].conj()
            };
        }
    }
    Tensor::from_vec_col_major(vec![n, n], values).unwrap()
}

// Elementwise/reduction.
fn add_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.add(&tensor_f64(&[262_144], 1), &tensor_f64(&[262_144], 2))?);
    Ok(())
}
fn sub_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.sub(&tensor_f64(&[262_144], 1), &tensor_f64(&[262_144], 2))?);
    Ok(())
}
fn mul_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.mul(&tensor_f64(&[262_144], 1), &tensor_f64(&[262_144], 2))?);
    Ok(())
}
fn div_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.div(
        &tensor_f64(&[262_144], 1),
        &tensor_f64_positive(&[262_144], 2),
    )?);
    Ok(())
}
fn rem_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.rem(
        &tensor_f64(&[262_144], 1),
        &tensor_f64_positive(&[262_144], 2),
    )?);
    Ok(())
}
fn neg_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.neg(&tensor_f64(&[262_144], 1))?);
    Ok(())
}
fn abs_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.abs(&tensor_f64(&[262_144], 1))?);
    Ok(())
}
fn sign_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.sign(&tensor_f64(&[262_144], 1))?);
    Ok(())
}
fn maximum_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.maximum(&tensor_f64(&[262_144], 1), &tensor_f64(&[262_144], 2))?);
    Ok(())
}
fn minimum_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.minimum(&tensor_f64(&[262_144], 1), &tensor_f64(&[262_144], 2))?);
    Ok(())
}
fn compare_lt_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.compare(
        &tensor_f64(&[262_144], 1),
        &tensor_f64(&[262_144], 2),
        &CompareDir::Lt,
    )?);
    Ok(())
}
fn select_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let cond = Tensor::from_vec_col_major(vec![262_144], data_bool(262_144)).unwrap();
    consume(b.select(
        &cond,
        &tensor_f64(&[262_144], 1),
        &tensor_f64(&[262_144], 2),
    )?);
    Ok(())
}
fn clamp_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let lower = Tensor::from_vec_col_major(vec![262_144], vec![-0.5; 262_144]).unwrap();
    let upper = Tensor::from_vec_col_major(vec![262_144], vec![0.5; 262_144]).unwrap();
    consume(b.clamp(&tensor_f64(&[262_144], 1), &lower, &upper)?);
    Ok(())
}
fn exp_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.exp(&tensor_f64(&[262_144], 1))?);
    Ok(())
}
fn log_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.log(&tensor_f64_positive(&[262_144], 1))?);
    Ok(())
}
fn sin_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.sin(&tensor_f64(&[262_144], 1))?);
    Ok(())
}
fn cos_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.cos(&tensor_f64(&[262_144], 1))?);
    Ok(())
}
fn tanh_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.tanh(&tensor_f64(&[262_144], 1))?);
    Ok(())
}
fn sqrt_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.sqrt(&tensor_f64_positive(&[262_144], 1))?);
    Ok(())
}
fn rsqrt_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.rsqrt(&tensor_f64_positive(&[262_144], 1))?);
    Ok(())
}
fn pow_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let exponent = Tensor::from_vec_col_major(vec![262_144], vec![1.5; 262_144]).unwrap();
    consume(b.pow(&tensor_f64_positive(&[262_144], 1), &exponent)?);
    Ok(())
}
fn expm1_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.expm1(&tensor_f64(&[262_144], 1))?);
    Ok(())
}
fn log1p_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.log1p(&tensor_f64_positive(&[262_144], 1))?);
    Ok(())
}
fn chain_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let x = b.log1p(&tensor_f64_positive(&[262_144], 1))?;
    let y = b.exp(&x)?;
    consume(b.mul(&y, &tensor_f64(&[262_144], 2))?);
    Ok(())
}
fn reduce_sum_all_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.reduce_sum(&tensor_f64(&[256, 1024], 1), &[0, 1])?);
    Ok(())
}
fn reduce_prod_all_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let x = Tensor::from_vec_col_major(vec![256, 1024], vec![1.000001; 256 * 1024]).unwrap();
    consume(b.reduce_prod(&x, &[0, 1])?);
    Ok(())
}
fn reduce_max_axis0_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.reduce_max(&tensor_f64(&[256, 1024], 1), &[0])?);
    Ok(())
}
fn reduce_min_axis1_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.reduce_min(&tensor_f64(&[256, 1024], 1), &[1])?);
    Ok(())
}

// Indexing/layout.
fn gather_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let indices = Tensor::from_vec_col_major(vec![65_536], data_i64(65_536, 65_536)).unwrap();
    consume(b.gather(
        &tensor_f64(&[65_536], 1),
        &indices,
        &GatherConfig {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1],
        },
    )?);
    Ok(())
}
fn scatter_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let operand = Tensor::from_vec_col_major(vec![65_536], vec![0.0; 65_536]).unwrap();
    let indices = Tensor::from_vec_col_major(vec![65_536, 1], data_i64(65_536, 65_536)).unwrap();
    consume(b.scatter(
        &operand,
        &indices,
        &tensor_f64(&[65_536], 2),
        &ScatterConfig {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        },
    )?);
    Ok(())
}
fn slice_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.slice(
        &tensor_f64(&[65_536], 1),
        &SliceConfig {
            starts: vec![1024],
            limits: vec![65_536 - 1024],
            strides: vec![2],
        },
    )?);
    Ok(())
}
fn dynamic_slice_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let starts = Tensor::from_vec_col_major(vec![1], vec![1024_i64]).unwrap();
    consume(b.dynamic_slice(&tensor_f64(&[65_536], 1), &starts, &[32_768])?);
    Ok(())
}
fn dynamic_update_slice_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let starts = Tensor::from_vec_col_major(vec![1], vec![1024_i64]).unwrap();
    consume(b.dynamic_update_slice(
        &tensor_f64(&[65_536], 1),
        &tensor_f64(&[32_768], 2),
        &starts,
    )?);
    Ok(())
}
fn pad_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.pad(
        &tensor_f64(&[65_536], 1),
        &PadConfig {
            edge_padding_low: vec![128],
            edge_padding_high: vec![128],
            interior_padding: vec![0],
        },
    )?);
    Ok(())
}
fn concatenate_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let a = tensor_f64(&[32_768], 1);
    let c = tensor_f64(&[32_768], 2);
    consume(b.concatenate(&[&a, &c], 0)?);
    Ok(())
}
fn reverse_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.reverse(&tensor_f64(&[65_536], 1), &[0])?);
    Ok(())
}

// Linalg.
fn cholesky_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(spd(128, 1).cholesky(b)?);
    Ok(())
}
fn eig_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let (w, v) = well_conditioned(64, 1).eig(b)?;
    consume_many([w, v]);
    Ok(())
}
fn eigvals_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(well_conditioned(64, 1).eigvals(b)?);
    Ok(())
}
fn eigvalsh_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(spd(128, 1).eigvalsh(b)?);
    Ok(())
}
fn triangular_solve_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(lower_triangular(128, 1).triangular_solve(
        &tensor_f64(&[128, 16], 2),
        true,
        true,
        false,
        false,
        b,
    )?);
    Ok(())
}
fn det_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(well_conditioned(128, 1).det(b)?);
    Ok(())
}
fn slogdet_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let (s, l) = well_conditioned(128, 1).slogdet(b)?;
    consume_many([s, l]);
    Ok(())
}
fn inv_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(well_conditioned(128, 1).inv(b)?);
    Ok(())
}
fn pinv_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(tensor_f64(&[128, 64], 1).pinv(b)?);
    Ok(())
}
fn norm_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(tensor_f64(&[256, 256], 1).norm(None, Some(&[0, 1]), false, b)?);
    Ok(())
}
fn full_piv_lu_solve_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(well_conditioned(64, 1).full_piv_lu_solve(&tensor_f64(&[64, 8], 2), b)?);
    Ok(())
}

// Complex.
fn conj_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.conj(&tensor_c64(&[65_536], 1))?);
    Ok(())
}
fn mul_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.mul(&tensor_c64(&[65_536], 1), &tensor_c64(&[65_536], 2))?);
    Ok(())
}
fn div_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let rhs =
        Tensor::from_vec_col_major(vec![65_536], vec![Complex64::new(1.5, 0.25); 65_536]).unwrap();
    consume(b.div(&tensor_c64(&[65_536], 1), &rhs)?);
    Ok(())
}
fn exp_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.exp(&tensor_c64(&[65_536], 1))?);
    Ok(())
}
fn log_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let x =
        Tensor::from_vec_col_major(vec![65_536], vec![Complex64::new(1.5, 0.25); 65_536]).unwrap();
    consume(b.log(&x)?);
    Ok(())
}
fn dot_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.dot_general(
        &tensor_c64(&[128, 128], 1),
        &tensor_c64(&[128, 128], 2),
        &DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    )?);
    Ok(())
}
fn svd_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let (u, s, vt) = tensor_c64(&[32, 32], 1).svd(b)?;
    consume_many([u, s, vt]);
    Ok(())
}
fn qr_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let (q, r) = tensor_c64(&[32, 32], 1).qr(b)?;
    consume_many([q, r]);
    Ok(())
}
fn eig_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let (w, v) = tensor_c64(&[32, 32], 1).eig(b)?;
    consume_many([w, v]);
    Ok(())
}
fn solve_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(tensor_c64(&[32, 32], 1).solve(&tensor_c64(&[32, 4], 2), b)?);
    Ok(())
}
fn cholesky_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(hpd_c64(32, 1).cholesky(b)?);
    Ok(())
}
fn norm_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(tensor_c64(&[64, 64], 1).norm(None, Some(&[0, 1]), false, b)?);
    Ok(())
}
