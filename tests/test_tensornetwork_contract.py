#!/usr/bin/env python3
"""Unit tests for tree-guided tensor network contraction helpers."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import tensornetwork_contract as tnc  # noqa: E402


class ListBackend:
    def einsum(self, ixs, iy, tensors):
        del ixs, iy
        return sum(tensors)


def test_integer_label_mapping_matches_upstream() -> None:
    backend = tnc.IntegerLabelEinsumBackend(lambda expr, *t: expr)
    expr = backend.einsum([[1, 2], [2, 3]], [1, 3], [])
    assert expr == "AB,BC->AC"


def test_contract_tree_follows_tensorindex() -> None:
    tree = {
        "isleaf": False,
        "eins": {"ixs": [[1, 2], [2, 3]], "iy": [1, 3]},
        "args": [
            {"isleaf": True, "tensorindex": 1},
            {"isleaf": True, "tensorindex": 2},
        ],
    }
    inputs = [10, 20]
    backend = ListBackend()
    assert tnc.contract_tree(tree, inputs, backend) == 30


def test_upstream_json_is_present() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "extern/TensorNetworkBenchmarks/data/tensornetwork_permutation_optimized.json"
    assert source.exists(), "TensorNetworkBenchmarks submodule not initialized"
    data = tnc.load_tensor_network(source)
    assert len(data["inputs"]) == 550
    assert "tree" in data


if __name__ == "__main__":
    test_integer_label_mapping_matches_upstream()
    test_contract_tree_follows_tensorindex()
    test_upstream_json_is_present()
    print("ok")
