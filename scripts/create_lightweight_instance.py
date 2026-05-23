#!/usr/bin/env python3
"""
Create a lightweight benchmark instance by extracting a connected subgraph
from an existing instance and computing an optimized contraction path.

Usage:
    python create_lightweight_instance.py <input_json> <output_json> <num_tensors>

The script:
1. Parses the tensor network graph from the format string
2. Finds a connected subset via BFS (choosing the seed that minimizes free indices)
3. Produces a scalar-output (->)  einsum problem from the subset
4. Computes an optimized contraction path via opt_einsum
5. Exports both row-major and column-major metadata
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import opt_einsum


def parse_format_string(format_string: str) -> tuple[list[str], str]:
    """Split format string into input operand labels and output labels."""
    inputs_str, output_str = format_string.split("->")
    return inputs_str.split(","), output_str


def build_adjacency(inputs: list[str]) -> dict[int, set[int]]:
    """Build an adjacency graph: tensor i is adjacent to tensor j if they share an index."""
    index_to_tensors: dict[str, set[int]] = defaultdict(set)
    for i, operand in enumerate(inputs):
        for c in operand:
            index_to_tensors[c].add(i)

    adj: dict[int, set[int]] = defaultdict(set)
    for tensors in index_to_tensors.values():
        tensor_list = list(tensors)
        for a in tensor_list:
            for b in tensor_list:
                if a != b:
                    adj[a].add(b)
    return adj


def bfs_subset(adj: dict[int, set[int]], seed: int, target: int) -> set[int]:
    """BFS from seed, returning up to target connected tensors."""
    visited = {seed}
    queue = [seed]
    while queue and len(visited) < target:
        current = queue.pop(0)
        for neighbor in sorted(adj[current]):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if len(visited) >= target:
                    break
    return visited


def count_free_indices(subset: set[int], inputs: list[str]) -> int:
    """Count indices appearing exactly once in the subset (free indices)."""
    chars: list[str] = []
    for t in subset:
        chars.extend(list(inputs[t]))
    counts = Counter(chars)
    return sum(1 for cnt in counts.values() if cnt == 1)


def find_best_subset(
    inputs: list[str], adj: dict[int, set[int]], target: int
) -> set[int]:
    """Try every tensor as BFS seed; return subset with fewest free indices."""
    best_free = math.inf
    best_subset: set[int] = set()
    for seed in range(len(inputs)):
        subset = bfs_subset(adj, seed, target)
        if len(subset) < target:
            continue
        n_free = count_free_indices(subset, inputs)
        if n_free < best_free:
            best_free = n_free
            best_subset = subset
    print(f"Best BFS seed gives {best_free} free indices for {target} tensors")
    return best_subset


def convert_format_string_to_colmajor(format_string: str) -> str:
    """Reverse each operand's index labels (row-major -> column-major)."""
    inputs_str, output_str = format_string.split("->")
    inputs = inputs_str.split(",")
    reversed_inputs = [op[::-1] for op in inputs]
    reversed_output = output_str[::-1]
    return ",".join(reversed_inputs) + "->" + reversed_output


def compute_contraction_path(
    format_string: str, shapes: list[list[int]], dtype: str
) -> tuple[list[list[int]], dict]:
    """Use opt_einsum to find optimized contraction paths."""
    np_dtype = np.float64 if dtype == "float64" else np.complex128

    # Build fake operands with correct shapes for path computation
    operands = [np.empty(shape, dtype=np_dtype) for shape in shapes]

    # Use "dp" (dynamic programming) for up to ~30 tensors, "auto" otherwise
    strategy = "dp" if len(shapes) <= 30 else "auto"
    try:
        path, info = opt_einsum.contract_path(
            format_string, *operands, optimize=strategy
        )
        opt_path = [list(pair) for pair in path]
        opt_meta = {
            "log10_flops": round(math.log10(max(info.opt_cost, 1)), 4),
            "log2_size": round(
                math.log2(max(info.largest_intermediate, 1)), 4
            ),
        }
    except Exception:
        n = len(shapes)
        opt_path = [[0, 1]] * (n - 1)
        opt_meta = {"log10_flops": 0.0, "log2_size": 0.0}

    return opt_path, opt_meta


def create_lightweight_instance(
    input_file: Path, output_file: Path, num_tensors_target: int
) -> None:
    with open(input_file) as f:
        data = json.load(f)

    original_num = data["num_tensors"]
    if num_tensors_target >= original_num:
        print(f"Target {num_tensors_target} >= original {original_num}, copying as-is")
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return

    inputs, _output = parse_format_string(data["format_string"])
    adj = build_adjacency(inputs)

    # Find connected subset with minimal free indices
    subset = find_best_subset(inputs, adj, num_tensors_target)
    subset_sorted = sorted(subset)

    # Extract operand labels and shapes for the subset
    subset_inputs = [inputs[i] for i in subset_sorted]
    subset_shapes = [data["shapes"][i] for i in subset_sorted]

    # Output = scalar (all indices summed over)
    new_format_string = ",".join(subset_inputs) + "->"

    print(f"Subset format string length: {len(new_format_string)}")
    print(f"Computing optimized contraction path...")

    # Compute optimized paths
    opt_path, opt_meta = compute_contraction_path(
        new_format_string, subset_shapes, data["dtype"]
    )

    print(
        f"Path found: log10_flops={opt_meta['log10_flops']}, "
        f"log2_size={opt_meta['log2_size']}"
    )

    # Column-major conversion
    new_format_string_cm = convert_format_string_to_colmajor(new_format_string)
    subset_shapes_cm = [shape[::-1] for shape in subset_shapes]

    # Build output JSON
    new_data = {
        "name": f"tensornetwork_permutation_light_{num_tensors_target}",
        "format_string": new_format_string,
        "shapes": subset_shapes,
        "dtype": data["dtype"],
        "num_tensors": num_tensors_target,
        "paths": {
            "opt_size": {
                "path": opt_path,
                "log10_flops": opt_meta["log10_flops"],
                "log2_size": opt_meta["log2_size"],
            },
            "opt_flops": {
                "path": opt_path,
                "log10_flops": opt_meta["log10_flops"],
                "log2_size": opt_meta["log2_size"],
            },
        },
        "format_string_rowmajor": new_format_string,
        "format_string_colmajor": new_format_string_cm,
        "shapes_colmajor": subset_shapes_cm,
    }

    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"Created lightweight instance: {original_num} -> {num_tensors_target} tensors")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python create_lightweight_instance.py "
            "<input_json> <output_json> <num_tensors>"
        )
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    num_tensors = int(sys.argv[3])

    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)

    create_lightweight_instance(input_file, output_file, num_tensors)
