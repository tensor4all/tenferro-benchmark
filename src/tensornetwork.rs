//! Tree-guided tensor network contraction for TensorNetworkBenchmarks parity.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_einsum::{EagerEinsumExt, GraphCompilerEinsumExt};
use tenferro_runtime::{GraphCompiler, Tensor, TracedTensor};
use tenferro_tensor::TypedTensor;

pub const DEFAULT_FILL_VALUE: f32 = 0.840_896_4; // 0.5f32.powf(0.4)
pub const DEFAULT_BOND_DIM: usize = 2;

#[derive(Debug, Deserialize)]
pub struct TensorNetworkFile {
    pub inputs: Vec<Vec<usize>>,
    pub tree: TreeNode,
}

#[derive(Debug, Deserialize)]
pub struct TreeNode {
    pub isleaf: bool,
    #[serde(default)]
    pub tensorindex: Option<usize>,
    #[serde(default)]
    pub eins: Option<EinsSpec>,
    #[serde(default)]
    pub args: Option<Vec<TreeNode>>,
}

#[derive(Debug, Deserialize)]
pub struct EinsSpec {
    pub ixs: Vec<Vec<usize>>,
    pub iy: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct TensorNetworkSpec {
    pub source: PathBuf,
    pub bond_dim: usize,
    pub fill_value: f32,
}

impl TensorNetworkSpec {
    pub fn from_problem(problem: &serde_yaml::Value, project_root: &Path) -> Result<Self, String> {
        let tn = problem
            .get("tensor_network")
            .ok_or("missing tensor_network block")?;
        let source = tn
            .get("source")
            .and_then(|v| v.as_str())
            .ok_or("tensor_network.source must be a string")?;
        let bond_dim = tn
            .get("bond_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_BOND_DIM as u64) as usize;
        let fill_value = tn
            .get("fill_value")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(DEFAULT_FILL_VALUE);
        Ok(Self {
            source: resolve_source_path(project_root, source),
            bond_dim,
            fill_value,
        })
    }
}

pub fn resolve_source_path(project_root: &Path, source: &str) -> PathBuf {
    let path = PathBuf::from(source);
    if path.is_absolute() {
        path
    } else {
        project_root.join(path)
    }
}

pub fn load_tensor_network(path: &Path) -> Result<TensorNetworkFile, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
}

pub fn tensor_f32_col_major(shape: &[usize], fill_value: f32) -> Result<Tensor, String> {
    let len: usize = shape.iter().product();
    TypedTensor::from_vec_col_major(shape.to_vec(), vec![fill_value; len])
        .map(Tensor::F32)
        .map_err(|e| format!("{e}"))
}

pub fn build_cpu_eager_inputs(
    network: &TensorNetworkFile,
    spec: &TensorNetworkSpec,
    ctx: Arc<EagerRuntime>,
) -> Result<Vec<EagerTensor>, String> {
    network
        .inputs
        .iter()
        .map(|ixs| {
            let shape = vec![spec.bond_dim; ixs.len()];
            EagerTensor::from_tensor_in(tensor_f32_col_major(&shape, spec.fill_value)?, ctx.clone())
                .map_err(|e| format!("{e}"))
        })
        .collect()
}

pub fn contract_tree_eager(node: &TreeNode, inputs: &[EagerTensor]) -> Result<EagerTensor, String> {
    if node.isleaf {
        let index = node
            .tensorindex
            .ok_or("leaf missing tensorindex")?
            .checked_sub(1)
            .ok_or("tensorindex must be >= 1")?;
        return inputs
            .get(index)
            .cloned()
            .ok_or_else(|| format!("tensorindex {index} out of range"));
    }

    let eins = node.eins.as_ref().ok_or("internal node missing eins")?;
    let args = node.args.as_ref().ok_or("internal node missing args")?;
    let child_tensors: Vec<EagerTensor> = args
        .iter()
        .map(|child| contract_tree_eager(child, inputs))
        .collect::<Result<_, _>>()?;
    let refs: Vec<&EagerTensor> = child_tensors.iter().collect();
    let expr = integer_labels_to_expr(&eins.ixs, &eins.iy);
    refs.as_slice()
        .einsum(&expr)
        .map_err(|e| format!("einsum ({expr}): {e}"))
}

pub fn contract_tree_trace(
    node: &TreeNode,
    inputs: &[(TracedTensor, Tensor)],
) -> Result<TracedTensor, String> {
    if node.isleaf {
        let index = node
            .tensorindex
            .ok_or("leaf missing tensorindex")?
            .checked_sub(1)
            .ok_or("tensorindex must be >= 1")?;
        return inputs
            .get(index)
            .map(|(t, _)| t.clone())
            .ok_or_else(|| format!("tensorindex {index} out of range"));
    }

    let eins = node.eins.as_ref().ok_or("internal node missing eins")?;
    let args = node.args.as_ref().ok_or("internal node missing args")?;
    let child_tensors: Vec<TracedTensor> = args
        .iter()
        .map(|child| contract_tree_trace(child, inputs))
        .collect::<Result<_, _>>()?;
    let refs: Vec<&TracedTensor> = child_tensors.iter().collect();
    let expr = integer_labels_to_expr(&eins.ixs, &eins.iy);
    let mut compiler = GraphCompiler::new();
    compiler
        .einsum(&refs, &expr)
        .map_err(|e| format!("einsum ({expr}): {e}"))
}

pub fn scalar_f32(result: &EagerTensor) -> Result<f32, String> {
    let tensor = result.to_tensor().map_err(|e| format!("{e}"))?;
    scalar_f32_tensor(&tensor)
}

pub fn scalar_f32_tensor(tensor: &Tensor) -> Result<f32, String> {
    match tensor {
        Tensor::F32(t) => t
            .host_data()
            .map_err(|e| format!("{e}"))?
            .first()
            .copied()
            .ok_or_else(|| "empty tensor result".to_string()),
        other => Err(format!("expected F32 result, got {other:?}")),
    }
}

pub fn integer_labels_to_expr(ixs: &[Vec<usize>], iy: &[usize]) -> String {
    let mut unique: HashSet<usize> = HashSet::new();
    for ix in ixs {
        unique.extend(ix.iter().copied());
    }
    unique.extend(iy.iter().copied());
    let mut labels: Vec<usize> = unique.into_iter().collect();
    labels.sort_unstable();

    let letters: Vec<char> = (65u8..90).chain(97u8..122).map(|c| c as char).collect();
    assert!(
        labels.len() <= letters.len(),
        "too many unique labels for ASCII einsum mapping"
    );
    let labelmap: std::collections::HashMap<usize, char> = labels
        .into_iter()
        .enumerate()
        .map(|(i, label)| (label, letters[i]))
        .collect();

    let inputs = ixs
        .iter()
        .map(|ix| ix.iter().map(|label| labelmap[label]).collect::<String>())
        .collect::<Vec<_>>()
        .join(",");
    let output: String = iy.iter().map(|label| labelmap[label]).collect();
    format!("{inputs}->{output}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_integer_labels_to_ascii_einsum() {
        let expr = integer_labels_to_expr(&[vec![1, 2], vec![2, 3]], &[1, 3]);
        assert_eq!(expr, "AB,BC->AC");
    }
}
