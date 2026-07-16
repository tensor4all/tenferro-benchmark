#!/usr/bin/env python3
"""Classify einsum benchmark gaps before choosing optimization targets."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


PATH_MAX_INTERMEDIATE_WARN = 50_000_000
PATH_TOTAL_INTERMEDIATE_WARN = 200_000_000


@dataclass(frozen=True)
class ResultRow:
    strategy: str
    instance: str
    tenferro_trace_ms: float | None
    tenferro_eager_ms: float | None
    pytorch_ms: float | None

    @property
    def best_tenferro_ms(self) -> float | None:
        values = [
            value
            for value in (self.tenferro_trace_ms, self.tenferro_eager_ms)
            if value is not None
        ]
        return min(values) if values else None


@dataclass(frozen=True)
class PathAnalysis:
    steps: int
    dot_steps: int
    outer_steps: int
    broadcast_mul_steps: int
    max_rank: int
    max_intermediate_elements: int
    total_intermediate_elements: int

    @property
    def warns(self) -> bool:
        return (
            self.max_intermediate_elements >= PATH_MAX_INTERMEDIATE_WARN
            or self.total_intermediate_elements >= PATH_TOTAL_INTERMEDIATE_WARN
        )


def clean_markdown_eof(markdown: str) -> str:
    return markdown.rstrip() + "\n"


def parse_ms_cell(cell: str) -> float | None:
    cell = cell.strip().replace("**", "").replace("`", "")
    if not cell or cell == "-":
        return None
    return float(cell.split("±", 1)[0].strip())


def parse_report(path: Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    strategy: str | None = None
    headers: list[str] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        match = re.match(r"^#### Strategy:\s+(\w+)", line)
        if match:
            strategy = match.group(1)
            headers = []
            continue
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells:
            continue
        if cells[0] == "Instance":
            headers = cells
            continue
        if cells[0].startswith("---") or strategy is None or not headers:
            continue

        by_header = dict(zip(headers, cells))
        rows.append(
            ResultRow(
                strategy=strategy,
                instance=by_header["Instance"],
                tenferro_trace_ms=parse_ms_cell(
                    by_header.get("tenferro-rs trace mode (ms)", "")
                ),
                tenferro_eager_ms=parse_ms_cell(
                    by_header.get("tenferro-rs eager mode (ms)", "")
                ),
                pytorch_ms=parse_ms_cell(by_header.get("PyTorch Python (ms)", "")),
            )
        )

    return rows


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def intermediate_output_labels(
    labels: list[list[str]], final_output: list[str], lhs_index: int, rhs_index: int
) -> list[str]:
    outside_counts: Counter[str] = Counter()
    for index, item in enumerate(labels):
        if index not in (lhs_index, rhs_index):
            outside_counts.update(item)

    labels_in_order: list[str] = []
    for label in labels[lhs_index] + labels[rhs_index]:
        if label not in labels_in_order:
            labels_in_order.append(label)

    label_set = set(labels_in_order)
    output: list[str] = []
    for label in final_output:
        if label in label_set and label not in output:
            output.append(label)
    for label in labels_in_order:
        if outside_counts[label] > 0 and label not in output:
            output.append(label)
    return output


def parse_format(fmt: str) -> tuple[list[list[str]], list[str]]:
    inputs, output = fmt.split("->", 1)
    return [list(item) for item in inputs.split(",") if item], list(output)


def analyze_path(instance: dict, strategy: str) -> PathAnalysis | None:
    paths = instance.get("paths") or {}
    path_meta = paths.get(strategy) or {}
    path = path_meta.get("path") or []
    fmt = instance.get("format_string_colmajor")
    shapes = instance.get("shapes_colmajor")
    if not fmt or not shapes:
        return None

    labels, final_output = parse_format(fmt)
    current_shapes = [list(shape) for shape in shapes]
    dot_steps = 0
    outer_steps = 0
    broadcast_mul_steps = 0
    max_rank = 0
    max_intermediate_elements = 0
    total_intermediate_elements = 0

    for pair in path:
        lhs_index, rhs_index = pair
        if lhs_index > rhs_index:
            lhs_index, rhs_index = rhs_index, lhs_index
        lhs_labels = labels[lhs_index]
        rhs_labels = labels[rhs_index]
        output_labels = intermediate_output_labels(
            labels, final_output, lhs_index, rhs_index
        )
        common = set(lhs_labels) & set(rhs_labels)
        if any(label not in output_labels for label in common):
            dot_steps += 1
        elif common:
            broadcast_mul_steps += 1
        else:
            outer_steps += 1

        dim_by_label: dict[str, int] = {}
        for label, dim in zip(lhs_labels, current_shapes[lhs_index]):
            dim_by_label[label] = dim
        for label, dim in zip(rhs_labels, current_shapes[rhs_index]):
            existing = dim_by_label.get(label)
            if existing is not None and existing != dim:
                raise ValueError(
                    f"{instance.get('name')}: dimension mismatch for label {label}"
                )
            dim_by_label[label] = dim

        output_shape = [dim_by_label[label] for label in output_labels]
        output_elements = product(output_shape)
        max_rank = max(max_rank, len(output_shape))
        max_intermediate_elements = max(max_intermediate_elements, output_elements)
        total_intermediate_elements += output_elements

        labels.pop(rhs_index)
        labels.pop(lhs_index)
        current_shapes.pop(rhs_index)
        current_shapes.pop(lhs_index)
        labels.append(output_labels)
        current_shapes.append(output_shape)

    return PathAnalysis(
        steps=len(path),
        dot_steps=dot_steps,
        outer_steps=outer_steps,
        broadcast_mul_steps=broadcast_mul_steps,
        max_rank=max_rank,
        max_intermediate_elements=max_intermediate_elements,
        total_intermediate_elements=total_intermediate_elements,
    )


def load_instances(path: Path) -> dict[str, dict]:
    instances: dict[str, dict] = {}
    for item in sorted(path.glob("*.json")):
        data = json.loads(item.read_text())
        # data/instances/ also holds non-einsum pattern files (e.g.
        # permutation_patterns.json for cpu/permutation) that are valid JSON
        # but not a single-instance record; skip anything missing "name".
        if not isinstance(data, dict) or "name" not in data:
            continue
        instances[data["name"]] = data
    return instances


def classify(
    row: ResultRow,
    analysis: PathAnalysis | None,
    threshold: float,
    min_delta_ms: float,
) -> str:
    best = row.best_tenferro_ms
    if best is None or row.pytorch_ms is None:
        return "missing_baseline"
    if best / row.pytorch_ms <= threshold:
        return "ok_or_faster"
    if best - row.pytorch_ms < min_delta_ms:
        return "small_absolute_gap"
    if analysis is not None and analysis.warns:
        return "path_intermediate"
    return "kernel_or_executor"


def format_rows(
    result_rows: list[ResultRow],
    instances: dict[str, dict],
    threshold: float,
    min_delta_ms: float,
) -> str:
    lines = [
        "| strategy | instance | ratio | class | best_tenferro_ms | pytorch_ms | max_intermediate_elements | total_intermediate_elements |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    decorated = []
    for row in result_rows:
        best = row.best_tenferro_ms
        if best is None or row.pytorch_ms is None:
            continue
        instance = instances.get(row.instance)
        analysis = analyze_path(instance, row.strategy) if instance else None
        ratio = best / row.pytorch_ms
        decorated.append(
            (ratio, row, analysis, classify(row, analysis, threshold, min_delta_ms))
        )

    for ratio, row, analysis, classification in sorted(
        decorated, key=lambda item: item[0], reverse=True
    ):
        max_intermediate = analysis.max_intermediate_elements if analysis else 0
        total_intermediate = analysis.total_intermediate_elements if analysis else 0
        lines.append(
            "| {} | {} | {:.3f} | {} | {:.3f} | {:.3f} | {} | {} |".format(
                row.strategy,
                row.instance,
                ratio,
                classification,
                row.best_tenferro_ms,
                row.pytorch_ms,
                max_intermediate,
                total_intermediate,
            )
        )
    return clean_markdown_eof("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("result/mac-cpu/cpu/einsum.md"))
    parser.add_argument("--instances", type=Path, default=Path("data/instances"))
    parser.add_argument("--threshold", type=float, default=1.15)
    parser.add_argument(
        "--min-delta-ms",
        type=float,
        default=0.5,
        help="Minimum absolute tenferro-vs-PyTorch gap before a slow ratio is actionable.",
    )
    args = parser.parse_args()

    print(
        format_rows(
            parse_report(args.report),
            load_instances(args.instances),
            args.threshold,
            args.min_delta_ms,
        )
    )


if __name__ == "__main__":
    main()
