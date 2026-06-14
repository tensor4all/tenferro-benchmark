pub mod tensornetwork;

use std::fmt::Display;

use tenferro_einsum::{einsum_subscripts_with, ContractionTree, EinsumOptimize, Subscripts};
use tenferro_runtime::{CompilerOptions, DType, GraphCompiler, GraphProgram, OptimizerConfig, TracedTensor};

pub struct CompiledEinsum {
    pub program: GraphProgram,
    pub inputs: Vec<TracedTensor>,
}

pub fn compile_einsum(
    subs: &Subscripts,
    shapes: &[Vec<usize>],
    tree: &ContractionTree,
) -> Result<CompiledEinsum, String> {
    let inputs: Vec<TracedTensor> = shapes
        .iter()
        .map(|shape| TracedTensor::input_concrete_shape(DType::F64, shape))
        .collect();
    let input_refs: Vec<&TracedTensor> = inputs.iter().collect();
    let einsum_subscripts = subs.into();
    let pairs: Vec<(usize, usize)> = (0..tree.step_count())
        .map(|idx| tree.step_pair(idx).expect("step index is in 0..step_count"))
        .collect();
    let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    let owned_tree =
        ContractionTree::from_pairs(subs, &shape_refs, &pairs).map_err(|e| format!("{e}"))?;
    let mut compiler = GraphCompiler::with_compiler_options(compiler_options_from_env());
    let output = einsum_subscripts_with(
        &mut compiler,
        &input_refs,
        &einsum_subscripts,
        EinsumOptimize::Tree(owned_tree),
    )
    .map_err(|e| format!("{e}"))?;
    let input_specs: Vec<(&TracedTensor, DType, &[usize])> = inputs
        .iter()
        .zip(shapes.iter())
        .map(|(input, shape)| (input, DType::F64, shape.as_slice()))
        .collect();
    let program = compiler
        .compile_with_input_specs(&output, &input_specs)
        .map_err(|e| format!("{e}"))?;

    Ok(CompiledEinsum { program, inputs })
}

fn compiler_options_from_env() -> CompilerOptions {
    let mut optimizer = OptimizerConfig::default();
    if std::env::var("TENFERRO_OPT_DOT_DECOMPOSER")
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    {
        optimizer.dot_decomposer = true;
    }
    CompilerOptions { optimizer }
}

pub fn unwrap_eval_result<T, E>(
    result: std::thread::Result<Result<T, E>>,
    panic_message: &str,
) -> Result<T, String>
where
    E: Display,
{
    match result {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(err)) => Err(err.to_string()),
        Err(_) => Err(panic_message.to_string()),
    }
}
