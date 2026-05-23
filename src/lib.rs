use std::fmt::Display;

use computegraph::compile::compile;
use computegraph::fragment::FragmentBuilder;
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, ValRef};
use tenferro::compiler::compile_std_to_exec;
use tenferro::exec::ExecProgram;
use tenferro_einsum::{build_einsum_fragment, ContractionTree, Subscripts};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::DType;
use tenferro_tensor::Tensor;

pub struct CompiledEinsum {
    pub exec: ExecProgram,
    pub input_keys: Vec<TensorInputKey>,
}

pub fn compile_einsum(
    subs: &Subscripts,
    shapes: &[Vec<usize>],
    tree: &ContractionTree,
) -> Result<CompiledEinsum, String> {
    let n_inputs = subs.inputs.len();

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let input_vals: Vec<ValRef<StdTensorOp>> = (0..n_inputs)
        .map(|id| {
            let key = TensorInputKey::User { id: id as u64 };
            ValRef::Local(builder.add_input(key))
        })
        .collect();

    let result = build_einsum_fragment(&mut builder, tree, &input_vals, shapes)
        .map_err(|e| format!("{e}"))?;
    let local_id = match result {
        ValRef::Local(local_id) => local_id,
        ValRef::External(_) => {
            return Err("einsum returned external ref for multi-input contraction".into());
        }
    };
    builder.set_outputs(vec![local_id]);

    let fragment = std::sync::Arc::new(builder.build());
    let output_key = fragment.vals()[fragment.outputs()[0]].key.clone();

    let view = resolve(vec![fragment]);
    let graph = materialize_merge(&view, &[output_key]);
    let compiled = compile(&graph);

    // input_keys を先に抽出（compile_std_to_exec の shape マッピングに必要）
    let input_keys = graph
        .inputs
        .iter()
        .map(|key| match key {
            GlobalValKey::Input(key) => Ok(key.clone()),
            _ => Err(format!("expected Input key in graph inputs, got {key:?}")),
        })
        .collect::<Result<Vec<_>, _>>()?;

    // compile_std_to_exec に渡す dtype と shape を構築
    let input_dtypes: Vec<DType> = input_keys.iter().map(|_| DType::F64).collect();
    let input_shapes_dim: Vec<Vec<DimExpr>> = input_keys
        .iter()
        .map(|key| match key {
            TensorInputKey::User { id } => {
                let idx = *id as usize;
                Ok(shapes[idx].iter().map(|&d| DimExpr::Const(d)).collect())
            }
            other => Err(format!(
                "benchmark runner expected user input keys, got {other:?}"
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let exec = compile_std_to_exec(&compiled, &input_dtypes, &input_shapes_dim);

    Ok(CompiledEinsum { exec, input_keys })
}

pub fn reorder_user_operands(
    input_keys: &[TensorInputKey],
    operands: Vec<Tensor>,
) -> Result<Vec<Tensor>, String> {
    if input_keys.len() != operands.len() {
        return Err(format!(
            "operand count mismatch: graph expects {} inputs but runner created {}",
            input_keys.len(),
            operands.len()
        ));
    }

    let mut by_user: Vec<Option<Tensor>> = operands.into_iter().map(Some).collect();
    let mut ordered = Vec::with_capacity(input_keys.len());
    for key in input_keys {
        let idx = match key {
            TensorInputKey::User { id } => {
                usize::try_from(*id).map_err(|_| format!("input id {id} does not fit in usize"))?
            }
            other => {
                return Err(format!(
                    "benchmark runner expected user input keys, got {other:?}"
                ));
            }
        };

        let tensor = by_user
            .get_mut(idx)
            .ok_or_else(|| format!("input id {idx} is out of range for operand list"))?
            .take()
            .ok_or_else(|| format!("duplicate or missing operand for input id {idx}"))?;
        ordered.push(tensor);
    }

    if by_user.iter().any(Option::is_some) {
        return Err("compiled graph did not consume every user operand exactly once".into());
    }

    Ok(ordered)
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
