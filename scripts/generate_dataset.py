"""Select and export einsum benchmark instances as JSON metadata.

Filters instances from einsum_benchmark by:
  - log10[FLOPS] < 10
  - log2[SIZE] < 25
  - dtype: float64 or complex128
  - num_tensors <= 100

Exports both row-major (original) and column-major metadata per instance.
"""

import json
from pathlib import Path

import numpy as np
import einsum_benchmark

def condition_lm(name: str) -> bool:
    instance = einsum_benchmark.instances[name]
    opt_flops_path_meta = instance.paths.opt_flops
    if not opt_flops_path_meta.flops < 10:
        return False
    if not opt_flops_path_meta.size < 25:
        return False
    cond_A = "float64" in np.unique([t.dtype.name for t in instance.tensors])
    cond_B = "complex128" in np.unique([t.dtype.name for t in instance.tensors])
    if not (cond_A or cond_B):
        return False
    if not len(instance.tensors) <= 100:
        return False
    return True

def condition_gm(name: str) -> bool:
    instance = einsum_benchmark.instances[name]
    opt_flops_path_meta = instance.paths.opt_flops
    if not opt_flops_path_meta.flops < 10:
        return False
    if not opt_flops_path_meta.size < 27:
        return False
    cond_A = "float64" in np.unique([t.dtype.name for t in instance.tensors])
    cond_B = "complex128" in np.unique([t.dtype.name for t in instance.tensors])
    if not (cond_A or cond_B):
        return False
    if not len(instance.tensors) <= 200:
        return False
    return True

def condition_str(name: str) -> bool:
    instance = einsum_benchmark.instances[name]
    opt_flops_path_meta = instance.paths.opt_flops
    if not opt_flops_path_meta.flops < 11:
        return False
    if not opt_flops_path_meta.size < 26:
        return False
    cond_A = "float64" in np.unique([t.dtype.name for t in instance.tensors])
    cond_B = "complex128" in np.unique([t.dtype.name for t in instance.tensors])
    if not (cond_A or cond_B):
        return False
    if not len(instance.tensors) <= 200:
        return False
    return True

def select_small_dataset(names: list[str]) -> list[str]:
    selected_names = []
    for name in names:
        if name.startswith("lm_"):
            if condition_lm(name):
                selected_names.append(name)
        elif name.startswith("gm_"):
            if condition_gm(name):
                selected_names.append(name)
        elif name.startswith("str_"):
            if condition_str(name):
                selected_names.append(name)
        elif name.startswith("rnd_mixed_"):
            """
            TODO:
            RND_MIXED instances are not supported yet in tenferro-einsum.
            """
            continue
        else:
            instance = einsum_benchmark.instances[name]
            cond_A = "float64" in np.unique([t.dtype.name for t in instance.tensors])
            cond_B = "complex128" in np.unique([t.dtype.name for t in instance.tensors])
            if not (cond_A or cond_B):
                continue
            if not len(instance.tensors) <= 100:
                continue
            selected_names.append(name)
    return selected_names


def build_instance_metadata(instance) -> dict:
    """Build metadata dict for a single einsum_benchmark instance (row-major, as-is)."""
    shapes = [list(t.shape) for t in instance.tensors]
    dtypes = [t.dtype.name for t in instance.tensors]
    dtype = np.unique(dtypes).tolist().pop()

    opt_size = instance.paths.opt_size
    opt_flops = instance.paths.opt_flops

    return {
        "name": instance.name,
        "format_string": instance.format_string,
        "shapes": shapes,
        "dtype": dtype,
        "num_tensors": len(instance.tensors),
        "paths": {
            "opt_size": {
                "path": [list(pair) for pair in opt_size.path],
                "log2_size": round(opt_size.size, 4),
                "log10_flops": round(opt_size.flops, 4),
            },
            "opt_flops": {
                "path": [list(pair) for pair in opt_flops.path],
                "log2_size": round(opt_flops.size, 4),
                "log10_flops": round(opt_flops.flops, 4),
            },
        },
    }


def convert_format_string_to_colmajor(format_string: str) -> str:
    """Convert einsum format_string from row-major to column-major convention.

    Row-major "ij,jk->ik" becomes column-major "ji,kj->ki".
    Each operand's index labels are reversed.
    """
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
    else:
        return ",".join(reversed_inputs)


def add_column_major_meta(meta: dict) -> dict:
    """Add column-major metadata to an instance metadata dict.

    - Reverse each tensor's shape
    - Reverse each operand's index labels in format_string
    - Contraction paths remain unchanged (they reference tensor indices, not dimensions)
    """
    return {
        **meta,
        "format_string_rowmajor": meta["format_string"],
        "format_string_colmajor": convert_format_string_to_colmajor(
            meta["format_string"]
        ),
        "shapes_colmajor": [shape[::-1] for shape in meta["shapes"]],
    }


def main():
    names = [instance.name for instance in einsum_benchmark.instances]
    selected_names = select_small_dataset(names)

    # Build metadata for all selected instances
    instances_metadata = []
    for name in selected_names:
        instance = einsum_benchmark.instances[name]
        meta = build_instance_metadata(instance)
        meta_col = add_column_major_meta(meta)
        instances_metadata.append(meta_col)

    print(f"Selected {len(instances_metadata)} instances:")
    for m in instances_metadata:
        print(
            f"  - {m['name']} (dtype={m['dtype']}, tensors={m['num_tensors']}, "
            f"log10[FLOPS]={m['paths']['opt_flops']['log10_flops']}, "
            f"log2[SIZE]={m['paths']['opt_flops']['log2_size']})"
        )

    # Export to JSON
    output_dir = Path(__file__).resolve().parent.parent / "data" / "instances"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual instance files
    for m in instances_metadata:
        output_path = output_dir / f"{m['name']}.json"
        with open(output_path, "w") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
