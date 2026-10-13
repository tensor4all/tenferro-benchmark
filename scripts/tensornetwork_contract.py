"""Tree-guided tensor network contraction (TensorNetworkBenchmarks parity).

Loads the JSON format used by extern/TensorNetworkBenchmarks and executes the
pre-optimized contraction tree step by step, matching benchmark_pytorch.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, Sequence, TypeVar

DEFAULT_FILL_VALUE = 0.5**0.4
DEFAULT_BOND_DIM = 2


class EinsumBackend(Protocol):
    def einsum(
        self, ixs: Sequence[Sequence[int]], iy: Sequence[int], tensors: Sequence[Any]
    ) -> Any: ...


T = TypeVar("T")


def _jax_binary_contract(ixs, iy):
    """Lower ordinary binary contractions directly, with no einsum optimizer.

    Shared output labels are batch axes. Other shared labels are contracted;
    the final transpose restores the tree's declared output ordering.
    """
    if len(ixs) != 2 or any(len(ix) != len(set(ix)) for ix in ixs):
        return None
    left, right = map(list, ixs)
    output = list(iy)
    shared = set(left) & set(right)
    if set(left) | set(right) != set(output) | shared:
        return None  # unilateral reductions require a different lowering
    batches = [label for label in output if label in shared]
    summed = [label for label in left if label in shared and label not in output]
    remaining_left = [label for label in left if label not in shared]
    remaining_right = [label for label in right if label not in shared]
    native = batches + remaining_left + remaining_right
    permutation = tuple(native.index(label) for label in output)
    dimensions = ((tuple(left.index(i) for i in summed), tuple(right.index(i) for i in summed)),
                  (tuple(left.index(i) for i in batches), tuple(right.index(i) for i in batches)))

    def execute(a, b):
        import jax
        result = jax.lax.dot_general(a, b, dimensions, precision=jax.lax.Precision.HIGHEST)
        return result if permutation == tuple(range(len(native))) else result.transpose(permutation)

    return execute


def resolve_source_path(project_root: Path, source: str) -> Path:
    path = Path(source)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_tensor_network(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = json.load(fh)
    if "tree" not in data or "inputs" not in data:
        raise ValueError(f"{path}: expected 'tree' and 'inputs' fields")
    return data


def labels_to_einsum(
    ixs: Sequence[Sequence[int]], iy: Sequence[int]
) -> tuple[list[list[int]], list[int]]:
    return [list(ix) for ix in ixs], list(iy)


def contract_tree(
    tree: dict[str, Any], inputs: Sequence[T], backend: EinsumBackend
) -> T:
    """Contract using the JSON tree and pre-built input tensors."""

    def contract_recur(node: dict[str, Any]) -> T:
        if node.get("isleaf"):
            index = int(node["tensorindex"]) - 1
            return inputs[index]
        child_tensors = [contract_recur(arg) for arg in node["args"]]
        eins = node["eins"]
        ixs, iy = labels_to_einsum(eins["ixs"], eins["iy"])
        return backend.einsum(ixs, iy, child_tensors)

    return contract_recur(tree)


def prepare_tree(tree: dict[str, Any], inputs: Sequence[Any], backend_name: str):
    """Prepare each contraction and label mapping before operation timing."""
    import opt_einsum as oe

    def prepare(node):
        if node.get("isleaf"):
            index = int(node["tensorindex"]) - 1
            return (lambda values: values[index]), tuple(inputs[index].shape)
        children = [prepare(child) for child in node["args"]]
        ixs, iy = node["eins"]["ixs"], node["eins"]["iy"]
        labels = sorted({label for ix in ixs for label in ix} | set(iy))
        names = {label: oe.get_symbol(i) for i, label in enumerate(labels)}
        expression = (
            ",".join("".join(names[i] for i in ix) for ix in ixs)
            + "->"
            + "".join(names[i] for i in iy)
        )
        shapes = [shape for _, shape in children]
        contract = oe.contract_expression(expression, *shapes, optimize="greedy")
        binary = _jax_binary_contract(ixs, iy) if backend_name == "jax" else None
        sizes = {
            label: size
            for ix, shape in zip(ixs, shapes)
            for label, size in zip(ix, shape)
        }

        def execute(values):
            operands = [run(values) for run, _ in children]
            if binary is not None:
                return binary(*operands)
            return contract(*operands, backend=backend_name)

        return execute, tuple(sizes[i] for i in iy)

    return prepare(tree)[0]


def build_input_shapes(
    inputs: Sequence[Sequence[int]], bond_dim: int = DEFAULT_BOND_DIM
) -> list[tuple[int, ...]]:
    return [tuple(bond_dim for _ in ix) for ix in inputs]


def build_fill_value(fill_value: float | None = None) -> float:
    return DEFAULT_FILL_VALUE if fill_value is None else float(fill_value)


def constant_tree_reference(network, bond_dim, fill_value):
    """Exact constant-input tree result, without allocating intermediates.

    Each entry is a product of child constants multiplied by the number of
    summed assignments. Logarithms avoid intermediate overflow/underflow. The
    input constant is rounded to F32 exactly as in the maintained CUDA fixture.
    This oracle applies only to the declared constant-fill workload.
    """
    import numpy as np
    value = float(np.float32(fill_value))
    if value <= 0 or bond_dim < 1:
        raise ValueError("constant-tree reference requires positive fill and dimension")

    def evaluate(node):
        if node.get("isleaf"):
            axes = network["inputs"][int(node["tensorindex"]) - 1]
            return float(np.log2(value)), len(axes)
        children = [evaluate(child) for child in node["args"]]
        ixs, iy = node["eins"]["ixs"], node["eins"]["iy"]
        if len(children) != len(ixs) or any(rank != len(ix) for (_, rank), ix in zip(children, ixs)):
            raise ValueError("constant-tree reference: inconsistent child ranks")
        labels = {label for ix in ixs for label in ix}
        if len(iy) != len(set(iy)) or not set(iy) <= labels:
            raise ValueError("constant-tree reference: invalid output labels")
        summed = len(labels - set(iy))
        return sum(log_value for log_value, _ in children) + summed * float(np.log2(bond_dim)), len(iy)

    logarithm, rank = evaluate(network["tree"])
    return np.full((bond_dim,) * rank, np.exp2(logarithm), dtype=np.float64)


def prepare_tree_jax_kernels(tree, inputs, compiler_options=None):
    """Compile and initialize each tree contraction, preserving the full tree.

    Planning, compilation and the representative initialization execution are
    outside timing. Unlike whole-tree JIT, execution dispatches the prepared
    kernels through the Python tree; callers must label these paths separately.
    """
    import jax
    import opt_einsum as oe
    prepared_nodes = [0]

    def prepare(node):
        if node.get("isleaf"):
            index = int(node["tensorindex"]) - 1
            return (lambda values: values[index]), inputs[index]
        children = [prepare(child) for child in node["args"]]
        ixs, iy = node["eins"]["ixs"], node["eins"]["iy"]
        labels = sorted({label for ix in ixs for label in ix} | set(iy))
        names = {label: oe.get_symbol(i) for i, label in enumerate(labels)}
        expr = ",".join("".join(names[label] for label in ix) for ix in ixs) + "->" + "".join(names[label] for label in iy)
        operands = tuple(value for _, value in children)
        expression = oe.contract_expression(expr, *(value.shape for value in operands), optimize="greedy")
        binary = _jax_binary_contract(ixs, iy)
        operation = binary or (lambda *values: expression(*values, backend="jax"))
        compiled = jax.jit(operation).lower(*operands).compile(compiler_options=compiler_options or {})
        representative = jax.block_until_ready(compiled(*operands))
        prepared_nodes[0] += 1
        if prepared_nodes[0] % 32 == 0:
            print(f"Prepared JAX tree kernels: {prepared_nodes[0]}", flush=True)
        executors = tuple(run for run, _ in children)

        def execute(values):
            return compiled(*(child(values) for child in executors))

        return execute, representative

    execute, representative = prepare(tree)
    del representative
    return execute


class IntegerLabelEinsumBackend:
    """Map integer index labels to ASCII einsum letters."""

    def __init__(self, einsum_fn):
        self._einsum_fn = einsum_fn

    def einsum(
        self, ixs: Sequence[Sequence[int]], iy: Sequence[int], tensors: Sequence[Any]
    ) -> Any:
        unique = sorted(set(sum((list(ix) for ix in ixs), []) + list(iy)))
        letters = list(range(65, 90)) + list(range(97, 122))
        if len(unique) > len(letters):
            raise ValueError(
                f"too many unique labels ({len(unique)}) for ASCII einsum mapping"
            )
        labelmap = {label: chr(letters[i]) for i, label in enumerate(unique)}
        expr_inputs = ["".join(labelmap[label] for label in ix) for ix in ixs]
        expr_output = "".join(labelmap[label] for label in iy)
        expr = ",".join(expr_inputs) + "->" + expr_output
        return self._einsum_fn(expr, *tensors)


def scalar_value(result: Any) -> float:
    if hasattr(result, "item"):
        return float(result.item())
    if hasattr(result, "reshape"):
        flat = result.reshape(-1)
        if hasattr(flat, "__getitem__"):
            return float(flat[0])
    return float(result)
