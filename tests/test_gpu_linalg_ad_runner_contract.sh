#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# A framework VJP row must include the forward pass, because the tenferro trace
# row it sits beside does: `vjp(fn, x)` builds the pullback from a fresh forward
# pass per sample, while hoisting it into the setup measures only the cached
# backward apply. This check reads the AST instead of an indentation depth, so
# moving the builder to module scope (which broke the previous regex) cannot
# silence it.
python3 - <<'PY'
import ast
import sys
from pathlib import Path

PATH = Path("scripts/benchmark_gpu_linalg_ad.py")
tree = ast.parse(PATH.read_text())


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def builder(name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SystemExit(f"{PATH}: no {name}() to check")


def calls(node, name):
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and dotted(child.func) == name
    ]


def closure_span(node, kind):
    return (node.lineno, node.end_lineno)


def inside(call, span):
    return span[0] <= call.lineno <= span[1]


problems = []

# PyTorch: the vjp call lives in `runner`, the callable the sampler times.
torch_builder = builder("_torch_framework")
runners = [n for n in ast.walk(torch_builder)
           if isinstance(n, ast.FunctionDef) and n.name == "runner"]
torch_calls = calls(torch_builder, "torch.func.vjp")
if not torch_calls:
    problems.append("_torch_framework builds no torch.func.vjp call")
elif not runners:
    problems.append("_torch_framework has no runner() closure")
else:
    span = closure_span(runners[0], ast.FunctionDef)
    for call in torch_calls:
        if not inside(call, span):
            problems.append(
                f"line {call.lineno}: torch.func.vjp is outside runner(); "
                "the row would time the cached backward apply only"
            )

# JAX: the vjp call lives in the closure that is compiled and run per sample.
jax_builder = builder("_jax_framework")
jax_calls = calls(jax_builder, "jax.vjp")
if not jax_calls:
    problems.append("_jax_framework builds no jax.vjp call")
for call in jax_calls:
    # `operation` is a lambda on the jvp branch and a nested def on the vjp one.
    closures = [
        n for n in ast.walk(jax_builder)
        if isinstance(n, (ast.Lambda, ast.FunctionDef))
        and n is not jax_builder  # the builder's own span contains everything
        and inside(call, closure_span(n, ast.AST))
    ]
    if not closures:
        problems.append(
            f"line {call.lineno}: jax.vjp is not inside a closure, so it runs once "
            "at setup instead of once per sample"
        )

if problems:
    print("\n".join(problems), file=sys.stderr)
    raise SystemExit(1)
print("gpu_linalg_ad_runner_contract: ok")
PY

echo "gpu linalg AD runner contract passed"
