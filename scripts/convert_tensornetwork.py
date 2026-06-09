"""Convert TensorNetworkBenchmarks JSON format to benchmark suite format.

Reads tensornetwork_permutation_optimized.json (tree + inputs) and outputs
the shared JSON format used by tenferro-einsum-benchmark and strided-rs-benchmark-suite
(format_string, shapes, paths, format_string_colmajor, shapes_colmajor, etc.).

TensorNetworkBenchmarks format:
  - tree: nested binary contraction tree with eins.ixs, eins.iy, tensorindex for leaves
  - inputs: list of index configs; inputs[i] = indices for tensor i+1 (1-based)
  - Tensors use uniform size 2 per dimension (benchmark convention)

Output format (shared):
  - format_string: "ab,cd,...->0" (einsum notation)
  - shapes: list of shapes per tensor
  - paths: opt_size/opt_flops with path (contraction pairs), log2_size, log10_flops
  - format_string_colmajor, shapes_colmajor
"""

import json
from pathlib import Path


def extract_num_input_tensors(tree: dict) -> int:
    """Count leaf nodes (input tensors) in the tree."""
    if tree.get("isleaf"):
        return 1
    return sum(extract_num_input_tensors(a) for a in tree["args"])


def extract_path_from_tree(tree: dict, inputs: list) -> list[list[int]]:
    """Extract contraction path from tree via post-order traversal.

    Simulates the opt_einsum path convention: each step [i,j] contracts
    tensors at positions i,j in the current list; remove higher index first,
    then lower; append result to end.
    """
    num_inputs = extract_num_input_tensors(tree)
    # nodes: list of (is_leaf, tensor_index_or_None)
    # We track positions; when we contract, we remove and append
    path: list[list[int]] = []

    def process(node) -> int:
        """Process node, return position of result in current list."""
        nonlocal path
        if node.get("isleaf"):
            # Leaf: tensor index is 0-based (tensorindex is 1-based)
            return node["tensorindex"] - 1

        # Internal: process children, then contract
        left_pos = process(node["args"][0])
        right_pos = process(node["args"][1])
        # Contract: remove higher index first, then lower; result at end
        i, j = min(left_pos, right_pos), max(left_pos, right_pos)
        path.append([i, j])
        # After removal: indices > j become j-1, indices > i become i-1, result at end
        # New result is at position (current_len - 2) = len(nodes) - 2
        # We need to return the position. Simulate: after removing j and i,
        # we have len-2 elements, then we append. So result is at len-2.
        # But we're not tracking the full list - we're just building the path.
        # The caller doesn't need the position for the final result.
        # For intermediate nodes, the "position" matters for the next contraction.
        # Let me simulate the list properly.
        return -1  # Will fix below

    # We need to simulate the list to get correct positions for nested contractions
    # Use a list of "ids": 0..n-1 for inputs, n, n+1, ... for intermediates
    nodes: list[int] = list(range(num_inputs))

    def process_with_list(node) -> int:
        """Process node; return the id of the result tensor."""
        nonlocal path, nodes
        if node.get("isleaf"):
            return node["tensorindex"] - 1

        left_id = process_with_list(node["args"][0])
        right_id = process_with_list(node["args"][1])
        # Find positions of left_id and right_id in nodes
        i = nodes.index(left_id)
        j = nodes.index(right_id)
        if i > j:
            i, j = j, i
        path.append([i, j])
        # Remove j first, then i; append new tensor
        nodes.pop(j)
        nodes.pop(i)
        new_id = num_inputs + len(path) - 1
        nodes.append(new_id)
        return new_id

    process_with_list(tree)
    return path


def _label_chars() -> list[int]:
    """Yield codepoints for index labels (enough for 220+ unique indices)."""
    # a-z, A-Z, then Greek α-ω, Α-Ω, then extended
    yield from range(97, 123)   # a-z
    yield from range(65, 91)    # A-Z
    yield from range(0x03B1, 0x03C9 + 1)   # Greek α-ω
    yield from range(0x0391, 0x03A9 + 1)   # Greek Α-Ω
    # Additional: 0-9, then more Unicode
    yield from range(48, 58)   # 0-9
    for i in range(0x0400, 0x0500):  # Cyrillic block
        yield i


def build_format_string(inputs: list, num_tensors: int) -> str:
    """Build einsum format string from inputs (index configs per tensor)."""
    # Collect all unique labels
    all_labels: set[int] = set()
    for i in range(num_tensors):
        for lbl in inputs[i]:
            all_labels.add(lbl)
    # Map to letters (a-z, A-Z, Greek, etc.)
    sorted_labels = sorted(all_labels)
    chars = list(_label_chars())
    if len(sorted_labels) > len(chars):
        raise ValueError(
            f"Too many unique indices ({len(sorted_labels)}), need <= {len(chars)}"
        )
    labelmap = {lbl: chr(chars[i]) for i, lbl in enumerate(sorted_labels)}

    parts = []
    for i in range(num_tensors):
        s = "".join(labelmap[lbl] for lbl in inputs[i])
        parts.append(s)
    # Output is scalar
    return ",".join(parts) + "->0"


def build_shapes(inputs: list, num_tensors: int, size: int = 2) -> list[list[int]]:
    """Build shapes from inputs. TensorNetworkBenchmarks uses uniform size 2."""
    return [[size] * len(inputs[i]) for i in range(num_tensors)]


def convert_format_string_to_colmajor(format_string: str) -> str:
    """Convert einsum format_string from row-major to column-major."""
    if "->" in format_string:
        inputs_str, output_str = format_string.split("->")
    else:
        inputs_str = format_string
        output_str = None

    inputs = inputs_str.split(",")
    reversed_inputs = [operand[::-1] for operand in inputs]

    if output_str is not None:
        reversed_output = output_str[::-1]
        return ",".join(reversed_inputs) + "->" + reversed_output
    return ",".join(reversed_inputs)


def add_column_major_meta(meta: dict) -> dict:
    """Add column-major metadata."""
    return {
        **meta,
        "format_string_rowmajor": meta["format_string"],
        "format_string_colmajor": convert_format_string_to_colmajor(meta["format_string"]),
        "shapes_colmajor": [shape[::-1] for shape in meta["shapes"]],
    }


def estimate_log2_size_and_flops(
    shapes: list[list[int]], path: list[list[int]], fallback_log2_size: float | None = None
) -> tuple[float, float]:
    """Estimate log2(max intermediate size) and log10(flops).

    For TensorNetworkBenchmarks permutation_optimized: README cites 2^33.2 complexity.
    Use fallback_log2_size when provided (e.g. 33.2).
    """
    from math import log2, log10

    if fallback_log2_size is not None:
        # log10_flops ≈ log10(2^log2_size) = log2_size * log10(2) ≈ 0.301 * log2_size
        # For 2^33.2: log10_flops ≈ 10
        log2_sz = fallback_log2_size
        log10_f = round(log2_sz * 0.301, 4)
        return round(log2_sz, 4), log10_f

    # Simplified simulation
    current = [list(s) for s in shapes]
    max_size = 0
    total_flops = 0

    for i, j in path:
        if i > j:
            i, j = j, i
        si = current[j]
        sj = current[i]
        current.pop(j)
        current.pop(i)
        size = 1
        for d in si:
            size *= d
        for d in sj:
            size *= d
        max_size = max(max_size, size)
        total_flops += size
        new_len = len(si) + len(sj) - 2  # assume 1 shared index for binary contraction
        current.append([2] * max(1, new_len))

    log2_sz = log2(max_size) if max_size > 0 else 0
    log10_f = log10(total_flops) if total_flops > 0 else 0
    return round(log2_sz, 4), round(log10_f, 4)


def convert(
    src_path: Path,
    output_path: Path,
    name: str = "tensornetwork_permutation_optimized",
    log2_size_override: float | None = None,
):
    """Convert TensorNetworkBenchmarks JSON to strided-rs format."""
    with open(src_path) as f:
        data = json.load(f)

    tree = data["tree"]
    inputs = data["inputs"]

    num_tensors = extract_num_input_tensors(tree)
    if num_tensors > len(inputs):
        raise ValueError(
            f"Tree has {num_tensors} leaves but inputs has only {len(inputs)} elements"
        )

    path = extract_path_from_tree(tree, inputs)
    format_string = build_format_string(inputs, num_tensors)
    shapes = build_shapes(inputs, num_tensors)

    log2_size, log10_flops = estimate_log2_size_and_flops(
        shapes, path, fallback_log2_size=log2_size_override
    )

    meta = {
        "name": name,
        "format_string": format_string,
        "shapes": shapes,
        "dtype": "float64",
        "num_tensors": num_tensors,
        "paths": {
            "opt_size": {
                "path": path,
                "log2_size": log2_size,
                "log10_flops": log10_flops,
            },
            "opt_flops": {
                "path": path,
                "log2_size": log2_size,
                "log10_flops": log10_flops,
            },
        },
    }

    meta = add_column_major_meta(meta)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Converted: {src_path} -> {output_path}")
    print(f"  num_tensors={num_tensors}, path_steps={len(path)}")
    print(f"  log2_size={log2_size}, log10_flops={log10_flops}")
    return meta


def main():
    project_dir = Path(__file__).resolve().parent.parent
    src = (
        project_dir
        / "extern"
        / "TensorNetworkBenchmarks"
        / "data"
        / "tensornetwork_permutation_optimized.json"
    )
    out_dir = project_dir / "data" / "instances"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tensornetwork_permutation_optimized.json"

    if not src.exists():
        print(f"Source not found: {src}")
        return

    # README: "contraction complexity 2^33.2" for permutation_optimized
    convert(src, out_path, log2_size_override=33.2)


if __name__ == "__main__":
    main()
