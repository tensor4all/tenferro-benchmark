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
    def einsum(self, ixs: Sequence[Sequence[int]], iy: Sequence[int], tensors: Sequence[Any]) -> Any: ...


T = TypeVar("T")


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


def labels_to_einsum(ixs: Sequence[Sequence[int]], iy: Sequence[int]) -> tuple[list[list[int]], list[int]]:
    return [list(ix) for ix in ixs], list(iy)


def contract_tree(tree: dict[str, Any], inputs: Sequence[T], backend: EinsumBackend) -> T:
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


def build_input_shapes(inputs: Sequence[Sequence[int]], bond_dim: int = DEFAULT_BOND_DIM) -> list[tuple[int, ...]]:
    return [tuple(bond_dim for _ in ix) for ix in inputs]


def build_fill_value(fill_value: float | None = None) -> float:
    return DEFAULT_FILL_VALUE if fill_value is None else float(fill_value)


class IntegerLabelEinsumBackend:
    """Map integer index labels to ASCII einsum letters."""

    def __init__(self, einsum_fn):
        self._einsum_fn = einsum_fn

    def einsum(self, ixs: Sequence[Sequence[int]], iy: Sequence[int], tensors: Sequence[Any]) -> Any:
        unique = sorted(set(sum((list(ix) for ix in ixs), []) + list(iy)))
        letters = list(range(65, 90)) + list(range(97, 122))
        if len(unique) > len(letters):
            raise ValueError(f"too many unique labels ({len(unique)}) for ASCII einsum mapping")
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
