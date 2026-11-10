#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# A framework VJP row must include the forward pass, because the tenferro trace
# row it sits beside does: `vjp(fn, x)` builds the pullback from a fresh forward
# pass per sample, while hoisting it into the setup measures only the cached
# backward apply. This reads the AST instead of an indentation depth, so moving
# the builders to module scope (which broke the previous regex) cannot silence
# it, and it resolves the callable the sampler actually calls -- the one a
# builder returns as `runner`, and the one handed to `jax.jit` -- rather than
# any nested function that happens to contain the call.
python3 - <<'PY'
import ast
import sys
from pathlib import Path

PATH = Path("scripts/benchmark_gpu_linalg_ad.py")
tree = ast.parse(PATH.read_text())


def dotted(node):
    """`torch.func.vjp` for an Attribute chain rooted at a Name, else None."""
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


def callables_named(node, name):
    """Every `def name` and `name = lambda ...` bound inside `node`.

    A builder binds the same name on more than one branch (`runner` is a lambda
    on the JVP branch and a `def` on the VJP one), so all of them are returned:
    the call only has to sit in the one that handles its phase.
    """
    found = []
    for child in ast.walk(node):
        if isinstance(child, ast.FunctionDef) and child.name == name:
            found.append(child)
        if isinstance(child, ast.Assign):
            targets = [t for t in child.targets if isinstance(t, ast.Name)]
            if any(t.id == name for t in targets) and isinstance(child.value, ast.Lambda):
                found.append(child.value)
    return found


def returned_runner_names(node):
    """Names a builder hands back under the `runner` key of its result dict."""
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
            for key, value in zip(child.value.keys, child.value.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "runner"
                    and isinstance(value, ast.Name)
                ):
                    names.add(value.id)
    return names


def jitted_names(node):
    """Names passed as the callable to `jax.jit`."""
    names = set()
    for call in calls(node, "jax.jit"):
        if call.args and isinstance(call.args[0], ast.Name):
            names.add(call.args[0].id)
    return names


def inside(call, closure):
    return closure is not None and closure.lineno <= call.lineno <= closure.end_lineno


problems = []


def require_inside(arm, vjp_name, builder_name, callable_source, what):
    """Every `vjp_name` call must sit in a callable taken from `callable_source`."""
    frame = builder(builder_name)
    vjp_calls = calls(frame, vjp_name)
    if not vjp_calls:
        problems.append(f"{builder_name} builds no {vjp_name} call")
        return
    names = callable_source(frame)
    if not names:
        problems.append(f"{builder_name}: no callable this row's sampler uses ({what})")
        return
    closures = {name: callables_named(frame, name) for name in names}
    missing = sorted(name for name, nodes in closures.items() if not nodes)
    if missing:
        problems.append(f"{builder_name}: {what} {missing} cannot be resolved")
        return
    for call in vjp_calls:
        if not any(inside(call, closure)
                   for nodes in closures.values() for closure in nodes):
            names_text = ", ".join(sorted(closures))
            problems.append(
                f"line {call.lineno}: {vjp_name} is not inside {names_text} "
                f"({what}); the row would time the cached backward apply only"
            )


# PyTorch: the callable the sampler calls is the one returned as `runner`.
require_inside(
    "torch",
    "torch.func.vjp",
    "_torch_framework",
    returned_runner_names,
    "the builder returns under `runner`",
)

# JAX: the callable the sampler calls is the one compiled by `jax.jit`.
require_inside(
    "jax",
    "jax.vjp",
    "_jax_framework",
    jitted_names,
    "the builder hands to `jax.jit`",
)

if problems:
    print("\n".join(problems), file=sys.stderr)
    raise SystemExit(1)
print("gpu_linalg_ad_runner_contract: ok")
PY

echo "gpu linalg AD runner contract passed"
