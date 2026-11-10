#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# A framework VJP row must include the forward pass, because the tenferro trace
# row it sits beside does: `vjp(fn, x)` builds the pullback from a fresh forward
# pass per sample, while hoisting it into the setup measures only the cached
# backward apply. This reads the AST instead of an indentation depth, so moving
# the builders to module scope (which broke the previous regex) cannot silence
# it, and it follows the callable the sampler actually calls -- the one a builder
# returns as `runner`, and the one handed to `jax.jit` -- so a setup helper that
# builds the pullback early and compiles a wrapper is rejected too.
#
# The syntax it understands is deliberately restricted: a builder must hand back
# the runner through a dict literal (directly, or through a variable that holds
# one) and pass `jax.jit` either the callable's name or a lambda that calls it.
# That is what these two builders do; another shape needs this check extended,
# not bypassed.
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


def bindings(frame, name):
    """Every binding of `name` in the builder, as (line, callable or None).

    A non-lambda assignment (`runner = cached`) is recorded with no callable: the
    name is rebound to something this check cannot follow, and that has to fail
    closed rather than fall back to the older definition.
    """
    found = []
    for child in ast.walk(frame):
        if isinstance(child, ast.FunctionDef) and child.name == name:
            found.append((child.lineno, child))
        if isinstance(child, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in child.targets):
                callable_node = child.value if isinstance(child.value, ast.Lambda) else None
                found.append((child.lineno, callable_node))
    return found


def dict_value_at(frame, mapping, key):
    """The Name a dict literal (or the variable holding one) maps `key` to."""
    for k, v in zip(mapping.keys, mapping.values):
        if isinstance(k, ast.Constant) and k.value == key and isinstance(v, ast.Name):
            return v.id
    return None


def runner_uses(frame):
    """(name, line) for every runner the builder hands to its caller."""
    uses = []
    for node in ast.walk(frame):
        if not isinstance(node, ast.Return):
            continue
        value = node.value
        if isinstance(value, ast.Dict):
            name = dict_value_at(frame, value, "runner")
            if name:
                uses.append((name, node.lineno))
        elif isinstance(value, ast.Name):
            # `result = {...}` then `return result`
            holder = [
                a for a in ast.walk(frame)
                if isinstance(a, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == value.id for t in a.targets)
                and isinstance(a.value, ast.Dict)
                and a.lineno < node.lineno
            ]
            if not holder:
                continue
            latest = max(holder, key=lambda a: a.lineno)
            name = dict_value_at(frame, latest.value, "runner")
            if name:
                uses.append((name, node.lineno))
    return uses


def jit_uses(frame):
    """(name, line) for every callable handed to `jax.jit`."""
    uses = []
    for call in calls(frame, "jax.jit"):
        if not call.args:
            continue
        arg = call.args[0]
        if isinstance(arg, ast.Name):
            uses.append((arg.id, call.lineno))
        elif isinstance(arg, ast.Lambda):
            inner = [n.func.id for n in ast.walk(arg)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            uses.extend((name, call.lineno) for name in inner)
    return uses


def inside(call, closure):
    return closure is not None and closure.lineno <= call.lineno <= closure.end_lineno


def check(label, vjp_name, builder_name, use_finder, what):
    frame = builder(builder_name)
    vjp_calls = calls(frame, vjp_name)
    if not vjp_calls:
        problems.append(f"{builder_name} builds no {vjp_name} call")
        return
    uses = use_finder(frame)
    if not uses:
        problems.append(f"{builder_name}: no callable this row's sampler uses ({what})")
        return
    for name, use_line in sorted(set(uses)):
        candidates = [b for b in bindings(frame, name) if b[0] < use_line]
        if not candidates:
            problems.append(f"{builder_name}: {name} has no binding before line {use_line}")
            continue
        # The binding in effect at the use site is the newest one before it: an
        # earlier `def runner` that still holds the VJP does not count once the
        # name has been rebound, and a rebinding to something that is not a
        # def/lambda cannot be followed, so it fails closed.
        active_line, active = max(candidates, key=lambda b: b[0])
        if active is None:
            problems.append(
                f"{builder_name}: {name} is rebound at line {active_line} to something "
                f"this check cannot follow; keep it a def/lambda or extend the check"
            )
            continue
        for call in vjp_calls:
            if not inside(call, active):
                problems.append(
                    f"line {call.lineno}: {vjp_name} is not inside {name}, the callable "
                    f"{what} (the {label} row would time the cached backward apply only)"
                )


problems = []

# PyTorch: the callable the sampler calls is the one returned as `runner`.
check("PyTorch", "torch.func.vjp", "_torch_framework", runner_uses,
      "the builder returns under `runner`")

# JAX: the callable the sampler calls is the one compiled by `jax.jit`.
check("JAX", "jax.vjp", "_jax_framework", jit_uses,
      "the builder hands to `jax.jit`")

if problems:
    print("\n".join(problems), file=sys.stderr)
    raise SystemExit(1)
print("gpu_linalg_ad_runner_contract: ok")
PY

echo "gpu linalg AD runner contract passed"
