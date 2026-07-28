pub mod tensornetwork;

use std::fmt::Display;

use tenferro_einsum::{ContractionTree, EinsumOptimize, Subscripts, TraceContextEinsumExt};
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{
    CompiledGraph, CompilerOptions, DType, GraphCompiler, OptimizerConfig, TraceContext,
};

pub struct CompiledEinsum {
    pub program: CompiledGraph,
    pub input_count: usize,
}

pub fn compile_einsum(
    subs: &Subscripts,
    shapes: &[Vec<usize>],
    tree: &ContractionTree,
) -> Result<CompiledEinsum, String> {
    let mut trace = TraceContext::new();
    let inputs = shapes
        .iter()
        .map(|shape| {
            trace.input(ProgramInputSpec::new(
                DType::F64,
                shape.iter().copied().map(Into::into),
            ))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{e}"))?;
    let einsum_subscripts = subs.into();
    let pairs: Vec<(usize, usize)> = (0..tree.step_count())
        .map(|idx| tree.step_pair(idx).expect("step index is in 0..step_count"))
        .collect();
    let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    let owned_tree =
        ContractionTree::from_pairs(subs, &shape_refs, &pairs).map_err(|e| format!("{e}"))?;
    let output = trace
        .einsum_subscripts_with(
            &inputs,
            &einsum_subscripts,
            EinsumOptimize::Tree(owned_tree),
        )
        .map_err(|e| format!("{e}"))?;
    let graph = trace.finish(&[output]).map_err(|e| format!("{e}"))?;
    let mut compiler = GraphCompiler::with_compiler_options(compiler_options_from_env());
    let program = compiler
        .compile_traced_graph(&graph)
        .map_err(|e| format!("{e}"))?;

    Ok(CompiledEinsum {
        program,
        input_count: inputs.len(),
    })
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
