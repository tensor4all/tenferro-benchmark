"""Logical fixtures and native framework operations for the CUDA API audit."""
import numpy as np
from scipy import special


def fixtures(case, dtype):
    op, n = case["op"], case["n"]
    op = op.removesuffix("_into").removesuffix("_read")
    shape = case.get("shape", [n])
    i = np.arange(n, dtype=np.uint64)
    raw = (((i * 2654435761 + 679) % 2001).astype(np.float64) / 2000 - .5).reshape(shape)
    x = raw.astype(dtype)
    if op in {"cholesky", "svdvals", "eigvalsh", "pinv", "inv", "det", "triangular_solve"}:
        # SPD close to identity, so determinant does not overflow at large n.
        matrix = raw.astype(np.float64)
        x = (np.eye(shape[0]) + matrix @ matrix.T / shape[0]).astype(dtype)
        a = x.astype(np.float64)
        inputs = [x]
        if op == "cholesky":
            y = np.linalg.cholesky(a)
        elif op == "svdvals":
            y = np.linalg.svd(a, compute_uv=False)
        elif op == "eigvalsh":
            y = np.linalg.eigvalsh(a)
        elif op == "pinv":
            # Tolerance tensors are operation inputs: construct and upload them
            # with the fixture, rather than in the timed scalar overload.
            inputs += [np.asarray(0, dtype=dtype), np.asarray(1e-5, dtype=dtype)]
            y = np.linalg.pinv(a, rcond=float(inputs[2]))
        elif op == "inv":
            y = np.linalg.inv(a)
        elif op == "det":
            y = np.linalg.det(a)
        else:
            from scipy.linalg import solve_triangular
            inputs = [np.tril(x), (raw[:, :8] + .1).astype(dtype)]
            y = solve_triangular(inputs[0].astype(np.float64), inputs[1].astype(np.float64), lower=True)
        return inputs, np.asarray(y, dtype=dtype)
    if op in {"log", "log1p", "sqrt", "rsqrt", "pow"}:
        x = (raw + 1.5).astype(dtype)
    if op == "reduce_prod":
        x = (1 + raw * .001).astype(dtype)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        x = (raw + 1j * raw[::-1]).astype(dtype)
    rhs = (1.5 - raw).astype(dtype)
    if op == "compare":
        rhs = np.roll(x, 1).copy()
        rhs.flat[::7] = x.flat[::7]
    inputs = [x]
    # Reference arithmetic in float64/complex128, rounded once to output dtype.
    a = x.astype(np.complex128 if np.iscomplexobj(x) else np.float64)
    b = rhs.astype(a.dtype)
    unary = {"neg": np.negative, "abs": np.abs, "sign": np.sign, "conj": np.conj,
             "exp": np.exp, "expm1": np.expm1, "log": np.log, "log1p": np.log1p,
             "erf": special.erf, "sin": np.sin, "cos": np.cos, "tanh": np.tanh,
             "sqrt": np.sqrt, "rsqrt": lambda v: 1/np.sqrt(v), "sigmoid": special.expit,
             "silu": lambda v: v * special.expit(v), "softplus": lambda v: np.logaddexp(v, 0),
             "gelu": lambda v: v * .5 * (1 + special.erf(v / np.sqrt(2))),
             "gelu_tanh": lambda v: v * .5 * (1 + np.tanh(np.sqrt(2/np.pi)*(v+.044715*v**3)))}
    binary = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide,
              "rem": np.fmod, "pow": np.power, "maximum": np.maximum, "minimum": np.minimum,
              "compare": getattr(np, {"eq": "equal", "lt": "less", "le": "less_equal", "gt": "greater", "ge": "greater_equal"}[case.get("compare_direction", "gt")])}
    if op in unary:
        y = unary[op](a)
    elif op in binary:
        inputs.append(rhs)
        y = binary[op](a, b)
    elif op == "where_select":
        mask = raw > 0
        inputs.extend([rhs, mask])
        y = np.where(mask, a, b)
    elif op == "clamp":
        inputs.extend([np.array(-.2, dtype=dtype), np.array(.3, dtype=dtype)])
        y = np.clip(a, inputs[1], inputs[2])
    elif op.startswith("reduce_"):
        which = op.removeprefix("reduce_")
        axis = None if case.get("full_reduction") else 1
        y = np.sum(a*a, axis=axis) if which == "sum_squares" else getattr(np, which)(a, axis=axis)
    elif op in {"softmax", "log_softmax", "masked_softmax", "masked_log_softmax"}:
        if op.startswith("masked_"):
            mask = raw > -.1
            mask[:, 0] = True  # exclude the intentionally different all-masked policy
            inputs.append(mask)
            a = np.where(mask, a, -np.inf)
        y = (special.log_softmax if "log" in op else special.softmax)(a, axis=1)
    elif op == "layer_norm":
        y = (a-a.mean(axis=1, keepdims=True))/np.sqrt(a.var(axis=1, keepdims=True)+1e-5)
    elif op == "rms_norm":
        y = a/np.sqrt(np.mean(a*a, axis=1, keepdims=True)+1e-5)
    elif op == "pad":
        y = np.pad(a, ((0, 0), (8, 8)))
    elif op == "slice":
        y = a[:, :a.shape[1]//2]
    elif op == "reverse":
        y = np.flip(a, axis=1)
    elif op == "concatenate":
        inputs.append(rhs)
        y = np.concatenate((a, b), axis=1)
    elif op in {"extract_diag", "embed_diag"}:
        y = np.diag(a)
    elif op in {"tril", "triu"}:
        y = getattr(np, op)(a)
    elif op == "take_along_axis":
        index = np.arange(a.shape[1]//2, dtype=np.int64)[None, :] * 2
        if case.get("variant") == "batched_indices":
            index = (index + np.arange(a.shape[0], dtype=np.int64)[:, None]) % a.shape[1]
        inputs.append(index)
        y = np.take_along_axis(a, index, axis=1)
    elif op == "scatter":
        index = np.arange(n//2, dtype=np.int64) * 2
        updates = rhs[:n//2]
        inputs.extend([index[:, None], updates])
        y = a.copy()
        y[index] += updates
    elif op == "cast":
        return inputs, x.astype(np.float64 if x.dtype == np.float32 else np.float32)
    elif op == "norm":
        y = np.linalg.norm(a)
    elif op in {"tensordot", "vdot", "norm_squared"}:
        if op != "norm_squared":
            inputs.append(rhs)
        y = np.tensordot(a, b, axes=1) if op == "tensordot" else (np.vdot(a, b) if op == "vdot" else np.vdot(a, a).real)
    else:
        raise ValueError(f"unknown fixture operation {op}")
    return inputs, y.astype(np.bool_ if op == "compare" else dtype)


def torch_operation(torch, case, inputs, *, verify_only=False):
    x, op = inputs[0], case["op"]
    if op.endswith("_into"):
        base = op.removesuffix("_into").removesuffix("_read")
        fn = torch.conj_physical if base == "conj" else getattr(torch, base)
        count = 1 if verify_only else max(1, min(4096, (256 << 20) // (x.numel()*x.element_size())))
        destinations = [torch.empty_like(x) for _ in range(count)]
        return lambda i: fn(*inputs, out=destinations[i])
    unary = {"neg": torch.neg, "abs": torch.abs, "sign": torch.sign, "conj": torch.conj_physical,
             "exp": torch.exp, "expm1": torch.expm1, "log": torch.log, "log1p": torch.log1p,
             "erf": torch.erf, "sin": torch.sin, "cos": torch.cos, "tanh": torch.tanh,
             "sqrt": torch.sqrt, "rsqrt": torch.rsqrt, "sigmoid": torch.sigmoid,
             "silu": torch.nn.functional.silu, "softplus": torch.nn.functional.softplus}
    binary = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div,
              "rem": torch.fmod, "pow": torch.pow, "maximum": torch.maximum,
              "minimum": torch.minimum, "compare": getattr(torch, case.get("compare_direction", "gt"))}
    if op in unary:
        return lambda: unary[op](x)
    if op in binary:
        return lambda: binary[op](x, inputs[1])
    if op == "where_select":
        return lambda: torch.where(inputs[2], x, inputs[1])
    if op == "clamp":
        return lambda: torch.clamp(x, min=inputs[1], max=inputs[2])
    if op in {"gelu", "gelu_tanh"}:
        return lambda: torch.nn.functional.gelu(x, approximate="tanh" if op.endswith("tanh") else "none")
    if op in {"softmax", "log_softmax"}:
        return lambda: getattr(torch.nn.functional, op)(x, dim=1)
    if op == "layer_norm":
        return lambda: torch.nn.functional.layer_norm(x, (x.shape[1],), eps=1e-5)
    if op == "rms_norm":
        return lambda: torch.nn.functional.rms_norm(x, (x.shape[1],), eps=1e-5)
    if op == "pad":
        return lambda: torch.nn.functional.pad(x, (8, 8))
    if op == "slice":
        return lambda: torch.slice_copy(x, 1, 0, x.shape[1]//2)
    if op == "reverse":
        return lambda: torch.flip(x, (1,))
    if op == "concatenate":
        return lambda: torch.cat(inputs, dim=1)
    if op == "extract_diag":
        return lambda: torch.diagonal_copy(x)
    if op == "embed_diag":
        return lambda: torch.diag_embed(x)
    if op in {"tril", "triu"}:
        return lambda: getattr(torch, op)(x)
    if op == "take_along_axis":
        return lambda: torch.take_along_dim(x, inputs[1], dim=1)
    if op == "scatter":
        index = inputs[1].reshape(-1)  # metadata preparation before the timer
        return lambda: torch.scatter_add(x, 0, index, inputs[2])
    if op == "cast":
        dtype = torch.float64 if x.dtype == torch.float32 else torch.float32
        return lambda: x.to(dtype=dtype)
    if op in {"cholesky", "svdvals", "eigvalsh", "inv", "det", "norm"}:
        return lambda: getattr(torch.linalg, op)(x)
    if op == "pinv":
        return lambda: torch.linalg.pinv(x, atol=inputs[1], rtol=inputs[2])
    if op == "triangular_solve":
        return lambda: torch.linalg.solve_triangular(x, inputs[1], upper=False)
    if op == "tensordot":
        return lambda: torch.tensordot(x, inputs[1], dims=1)
    if op == "vdot_read":
        return lambda: torch.vdot(x, inputs[1])
    if op == "norm_squared_read":
        return lambda: torch.vdot(x, x)
    if op.startswith("reduce_"):
        fn = {"reduce_sum": torch.sum, "reduce_prod": torch.prod, "reduce_max": torch.amax,
              "reduce_min": torch.amin, "reduce_mean": torch.mean}[op]
        return (lambda: fn(x)) if case.get("full_reduction") else (lambda: fn(x, dim=1))
    raise ValueError(f"missing native PyTorch API: {op}")


def jax_operation(jax, jnp, case):
    op = case["op"]
    unary = {"neg": jnp.negative, "abs": jnp.abs, "sign": jnp.sign, "conj": jnp.conj,
             "exp": jnp.exp, "expm1": jnp.expm1, "log": jnp.log, "log1p": jnp.log1p,
             "erf": jax.lax.erf, "sin": jnp.sin, "cos": jnp.cos, "tanh": jnp.tanh,
             "sqrt": jnp.sqrt, "rsqrt": jax.lax.rsqrt, "sigmoid": jax.nn.sigmoid,
             "silu": jax.nn.silu, "softplus": jax.nn.softplus}
    binary = {"add": jnp.add, "sub": jnp.subtract, "mul": jnp.multiply, "div": jnp.divide,
              "rem": jnp.fmod, "pow": jnp.power, "maximum": jnp.maximum,
              "minimum": jnp.minimum, "compare": getattr(jnp, {"eq": "equal", "lt": "less", "le": "less_equal", "gt": "greater", "ge": "greater_equal"}[case.get("compare_direction", "gt")])}
    if op in unary:
        return lambda x: unary[op](x)
    if op in binary:
        return lambda x, y: binary[op](x, y)
    if op == "where_select":
        return lambda x, y, mask: jnp.where(mask, x, y)
    if op == "clamp":
        return lambda x, lo, hi: jnp.clip(x, lo, hi)
    if op in {"gelu", "gelu_tanh"}:
        return lambda x: jax.nn.gelu(x, approximate=op.endswith("tanh"))
    if op in {"softmax", "log_softmax"}:
        return lambda x: getattr(jax.nn, op)(x, axis=1)
    if op in {"masked_softmax", "masked_log_softmax"}:
        fn = jax.nn.log_softmax if "log" in op else jax.nn.softmax
        return lambda x, mask: fn(x, axis=1, where=mask)
    if op == "layer_norm":
        return lambda x: jax.nn.standardize(x, axis=1, epsilon=1e-5)
    if op == "pad":
        return lambda x: jnp.pad(x, ((0, 0), (8, 8)))
    if op == "slice":
        return lambda x: jax.lax.slice_in_dim(x, 0, x.shape[1]//2, axis=1)
    if op == "reverse":
        return lambda x: jnp.flip(x, axis=1)
    if op == "concatenate":
        return lambda x, y: jnp.concatenate((x, y), axis=1)
    if op in {"extract_diag", "embed_diag"}:
        return lambda x: jnp.diag(x)
    if op in {"tril", "triu"}:
        return lambda x: getattr(jnp, op)(x)
    if op == "take_along_axis":
        return lambda x, index: jnp.take_along_axis(x, index, axis=1)
    if op == "scatter":
        config = jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))
        return lambda x, index, updates: jax.lax.scatter_add(x, index, updates, config, indices_are_sorted=True, unique_indices=True)
    if op == "cast":
        dtype = jnp.float64 if case["dtype"] == "f32" else jnp.float32
        return lambda x: x.astype(dtype)
    if op in {"cholesky", "eigvalsh", "inv", "det", "norm"}:
        return lambda x: getattr(jnp.linalg, op)(x)
    if op == "svdvals":
        return lambda x: jnp.linalg.svd(x, compute_uv=False)
    if op == "pinv":
        return lambda x, atol, rtol: jnp.linalg.pinv(x, rtol=rtol)
    if op == "triangular_solve":
        import jax.scipy.linalg
        return lambda x, rhs: jax.scipy.linalg.solve_triangular(x, rhs, lower=True)
    if op == "tensordot":
        return lambda x, y: jnp.tensordot(x, y, axes=1)
    if op == "vdot_read":
        return lambda x, y: jnp.vdot(x, y)
    if op == "norm_squared_read":
        return lambda x: jnp.vdot(x, x)
    if op.startswith("reduce_"):
        fn = getattr(jnp, op.removeprefix("reduce_"))
        axis = None if case.get("full_reduction") else 1
        return lambda x: fn(x, axis=axis)
    raise ValueError(f"missing native JAX API: {op}")
