"""Logical fixtures and native framework operations for the CUDA API audit."""
import numpy as np
from scipy import special


def fixtures(case, dtype):
    if case.get("read_op"):
        logical, expected = fixtures(dict(case, op=case["read_op"], read_op=""), dtype)
        return [value.T.copy() for value in logical], expected
    op, n = case["op"], case["n"]
    op = op.removesuffix("_into").removesuffix("_read")
    shape = case.get("shape", [n])
    i = np.arange(n, dtype=np.uint64)
    raw = (((i * 2654435761 + 679) % 2001).astype(np.float64) / 2000 - .5).reshape(shape)
    x = raw.astype(dtype)
    if np.issubdtype(np.dtype(dtype), np.integer):
        integer = ((i * 97 + 5) % 129).astype(np.int64).reshape(shape) - 64
        rhs = (((i * 73 + 3) % 17).astype(np.int64) + 1).reshape(shape)
        rhs *= np.where((i.reshape(shape) % 2) == 0, 1, -1)
        if case["dtype"] == "i64" and op in {"neg", "abs", "sign", "maximum", "minimum", "reduce_max", "reduce_min"}:
            integer += np.where(integer >= 0, 1 << 54, -(1 << 54))
            rhs += np.where(rhs > 0, 1 << 54, -(1 << 54))
        x, rhs = integer.astype(dtype), rhs.astype(dtype)
        unary = {"neg": np.negative, "abs": np.abs, "sign": np.sign}
        binary = {"add": np.add, "sub": np.subtract, "mul": np.multiply,
                  "maximum": np.maximum, "minimum": np.minimum}
        if op in unary:
            return [x], unary[op](x)
        if op in binary:
            return [x, rhs], binary[op](x, rhs)
        if op in {"div", "rem"}:
            quotient = (np.abs(x) // np.abs(rhs)) * np.sign(x) * np.sign(rhs)
            return [x, rhs], (quotient if op == "div" else x - quotient * rhs).astype(dtype)
        if op in {"reduce_sum", "reduce_max", "reduce_min"}:
            fn = getattr(np, op.removeprefix("reduce_"))
            result = fn(x, axis=1, dtype=x.dtype) if op == "reduce_sum" else fn(x, axis=1)
            return [x], result
        raise ValueError(f"unselected integer fixture operation {op}")
    if op.startswith("einsum_plan"):
        precision = np.complex128 if np.issubdtype(np.dtype(dtype), np.complexfloating) else np.float64
        rhs_shape = case["rhs_shape"]
        indices = np.arange(np.prod(rhs_shape),dtype=np.uint64).reshape(rhs_shape)
        raw_rhs = ((indices*1597334677+131)%2017).astype(np.float64)/2017-.5
        a = (raw+.7j*np.roll(raw,3,axis=0) if precision==np.complex128 else raw).astype(dtype)
        b = (raw_rhs+.3j*np.roll(raw_rhs,7,axis=1) if precision==np.complex128 else raw_rhs).astype(dtype)
        expected = a.astype(precision) @ b.astype(precision)
        inputs = [a,b]
        if "_accum" in op:
            rows,cols = expected.shape
            seed_values = np.arange(rows*cols).reshape(rows,cols)
            seed = (1.5+.2*np.sin(seed_values*.17)).astype(dtype)
            if precision==np.complex128:
                seed = (seed+.1j*np.cos(seed_values*.23)).astype(dtype)
            inputs.append(seed)
            expected = .5*expected+seed.astype(precision)
        if "strided" in op:
            inputs[:2] = [a.T.copy(),b.T.copy()]
        return inputs,expected.astype(dtype)
    if op in {"dot_general_axes", "dot_general_conj"}:
        precision = np.complex128 if np.issubdtype(np.dtype(dtype), np.complexfloating) else np.float64
        a = (raw + .7j*np.roll(raw, 3, axis=0) if precision == np.complex128 else raw).astype(dtype)
        rhs_shape = case["rhs_shape"]
        indices = np.arange(np.prod(rhs_shape), dtype=np.uint64).reshape(rhs_shape)
        raw_rhs = ((indices*1597334677 + 131) % 2017).astype(np.float64)/2017 - .5
        b = (raw_rhs + .3j*np.roll(raw_rhs, 7, axis=-1) if precision == np.complex128 else raw_rhs).astype(dtype)
        lhs = a.astype(precision).conj() if case.get("lhs_conj") else a.astype(precision)
        rhs = b.astype(precision).conj() if case.get("rhs_conj") else b.astype(precision)
        return [a,b], np.asarray(np.einsum(case["equation"],lhs,rhs,optimize=True),dtype=dtype)
    if op == "lstsq":
        precision = np.complex128 if np.issubdtype(np.dtype(dtype), np.complexfloating) else np.float64
        matrix = raw + 1j*raw[::-1] if precision == np.complex128 else raw
        a = (np.eye(*shape) + .05*matrix).astype(dtype)
        rhs_shape = [shape[0], case["rhs_cols"]]
        indices = np.arange(np.prod(rhs_shape), dtype=np.uint64).reshape(rhs_shape)
        rhs = (((indices*719 + 13) % 1103).astype(np.float64)/1103 - .5)
        if precision == np.complex128:
            rhs = rhs + 1j*np.roll(rhs, 3, axis=0)
        b = rhs.astype(dtype)
        solution = np.linalg.lstsq(a.astype(precision), b.astype(precision), rcond=None)[0]
        return [a, b], solution.astype(dtype)
    if op == "lu":
        from scipy.linalg import lu, lu_factor
        precision = np.complex128 if np.issubdtype(np.dtype(dtype), np.complexfloating) else np.float64
        matrix = raw + 1j*raw[::-1] if precision == np.complex128 else raw
        x = np.roll((np.eye(*shape) + .05*matrix).astype(dtype), 1, axis=0)
        p, l, u = lu(x.astype(precision))
        _, pivots = lu_factor(x.astype(precision))
        parity = -1 if np.count_nonzero(pivots != np.arange(len(pivots))) % 2 else 1
        # tenferro uses P A = L U; SciPy returns A = P L U.
        return [x], (p.T.astype(dtype), l.astype(dtype), u.astype(dtype), np.asarray(parity, dtype=dtype))
    if op.startswith("dot_general"):
        precision = np.complex128 if np.issubdtype(np.dtype(dtype), np.complexfloating) else np.float64
        a = (raw + 1j*raw[::-1] if precision == np.complex128 else raw).astype(dtype)
        b = (np.eye(shape[0]) + .1*np.roll(a, 1, axis=0)).astype(dtype)
        seed = (np.roll(a, 3, axis=1) + .2).astype(dtype)
        product = a.astype(precision) @ b.astype(precision)
        expected = .5*product + seed.astype(precision) if "_into_accum" in case["op"] else product
        return [a, b, seed], expected.astype(dtype)
    if op in {"householder_from_factors", "householder_append", "householder_r", "householder_q_columns"}:
        matrix = raw + 1j*raw[::-1] if np.issubdtype(np.dtype(dtype), np.complexfloating) else raw
        x = (np.eye(*shape) + .05*matrix).astype(dtype)
        precision = np.complex128 if np.iscomplexobj(x) else np.float64
        q, r = np.linalg.qr(x.astype(precision), mode="complete")
        k = min(shape)
        if op == "householder_r":
            return [x], r[:k].astype(dtype)
        if op == "householder_q_columns":
            start, end = case["column_range"]
            return [x], q[:, start:end].astype(dtype)
        if op == "householder_from_factors":
            q, r = q[:, :k].astype(dtype), r[:k].astype(dtype)
            x = (q.astype(precision) @ r.astype(precision)).astype(dtype)
            inputs = [x, q, r]
        else:
            split = shape[1]//2
            inputs = [x, x[:, :split].copy(), x[:, split:].copy()]
        h, tau = np.linalg.qr(x.astype(precision), mode="raw")
        return inputs, (h.T.astype(dtype), tau.astype(dtype))
    if op in {"svd_full", "slogdet", "householder_qr"}:
        real_dtype = np.empty((), dtype=dtype).real.dtype
        matrix = raw + 1j*raw[::-1] if np.issubdtype(np.dtype(dtype), np.complexfloating) else raw
        if op == "householder_qr":
            x = (np.eye(*shape) + .05*matrix).astype(dtype)
            h, tau = np.linalg.qr(x.astype(np.complex128 if np.iscomplexobj(x) else np.float64), mode="raw")
            return [x], (h.T.astype(dtype), tau.astype(dtype))
        if op == "svd_full":
            x = (np.eye(*shape) + .05*matrix).astype(dtype)
            values = np.linalg.svd(x.astype(np.complex128 if np.iscomplexobj(x) else np.float64), full_matrices=True)
            return [x], tuple(v.astype(real_dtype if i == 1 else dtype) for i, v in enumerate(values))
        x = (np.eye(shape[0]) + matrix @ matrix.conj().T / shape[0]).astype(dtype)
        if case.get("variant") == "negative":
            x[0] *= -1
        if case.get("variant") == "phase":
            x[0] *= np.exp(.37j)
        sign, logabs = np.linalg.slogdet(x.astype(np.complex128 if np.iscomplexobj(x) else np.float64))
        return [x], (np.asarray(sign, dtype=dtype), np.asarray(logabs, dtype=real_dtype))
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
    if op == "axpby_read_into_accum":
        precision = np.complex128 if np.iscomplexobj(x) else np.float64
        if "alpha" in case:
            rhs = (1.5 + .2*np.sin(raw*5) + .1*np.cos(np.roll(raw, 7)*3)).astype(dtype)
            if np.iscomplexobj(x):
                rhs = (rhs + .125j*np.roll(raw, 3)).astype(dtype)
        return [x, rhs], (case.get("beta", 1.)*rhs.astype(precision) + case.get("alpha", .5)*x.astype(precision)).astype(dtype)
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
    if op in {"scale_real", "scale_complex"}:
        factor = case["alpha"] + (1j*case["beta"] if op == "scale_complex" else 0)
        scalar = np.asarray(factor, dtype=dtype)
        inputs.append(scalar)
        y = a * scalar.astype(a.dtype)
    elif op in unary:
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
    elif op in {"layer_norm", "rms_norm"}:
        y = ((a-a.mean(axis=1, keepdims=True))/np.sqrt(a.var(axis=1, keepdims=True)+1e-5)
             if op == "layer_norm" else a/np.sqrt(np.mean(a*a, axis=1, keepdims=True)+1e-5))
        coordinate = np.arange(a.shape[1], dtype=np.float64)
        if case.get("normalization_weight"):
            weight = (.7 + .3*np.cos(coordinate*.37)).astype(dtype)
            inputs.append(weight)
            y = y * weight.astype(np.float64)
        if case.get("normalization_bias"):
            bias = (.2*np.sin(coordinate*.53)).astype(dtype)
            inputs.append(bias)
            y = y + bias.astype(np.float64)
    elif op == "pad":
        padding = case.get("padding", [(0, 0, 0), (8, 8, 0)])
        shape = tuple(lo + hi + size + max(size - 1, 0) * interior
                      for size, (lo, hi, interior) in zip(a.shape, padding))
        y = np.zeros(shape, dtype=a.dtype)
        # Independent coordinate oracle covers holes and cropping together.
        coordinates = np.indices(a.shape)
        destination = [lo + coordinates[axis] * (interior + 1)
                       for axis, (lo, hi, interior) in enumerate(padding)]
        valid = np.ones(a.shape, dtype=bool)
        for axis, coord in enumerate(destination):
            valid &= (coord >= 0) & (coord < shape[axis])
        y[tuple(coord[valid] for coord in destination)] = a[valid]
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
    elif op == "broadcast_in_dim":
        y = np.broadcast_to(a, case["output_shape"])
    elif op == "dynamic_slice":
        starts = np.asarray(case["starts"], dtype=np.int64)
        inputs.append(starts)
        clamped = np.minimum(np.maximum(starts, 0), np.asarray(a.shape)-case["slice_sizes"])
        y = a[tuple(slice(int(start), int(start)+size) for start, size in zip(clamped, case["slice_sizes"]))]
    elif op == "gather":
        axis = case["gather_axis"]
        indices = (np.arange(a.shape[axis]//2, dtype=np.int64)*2)[:, None]
        inputs.append(indices)
        y = np.take(a, indices[:, 0], axis=axis)
    elif op == "convert":
        return inputs, x.astype(np.float64)
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
    if case.get("read_op"):
        return torch_operation(torch, dict(case, op=case["read_op"], read_op=""),
                               [value.transpose(0, 1) for value in inputs], verify_only=verify_only)
    x, op = inputs[0], case["op"]
    if case["dtype"] in {"i32", "i64"} and op == "div":
        return lambda: torch.div(x, inputs[1], rounding_mode="trunc")
    if case["dtype"] in {"i32", "i64"} and op == "reduce_sum":
        return lambda: torch.sum(x, dim=1, dtype=x.dtype)
    if op.startswith("einsum_plan"):
        a,b = inputs[:2]
        if "strided" in op:
            a,b = a.transpose(0,1),b.transpose(0,1)
        if "_into" not in op:
            return lambda: torch.mm(a,b)
        bytes_per_output = a.shape[0]*b.shape[1]*a.element_size()
        count = 1 if verify_only else max(1,min(4096,(256<<20)//bytes_per_output))
        destinations = [torch.empty((a.shape[0],b.shape[1]),dtype=a.dtype,device=a.device) for _ in range(count)]
        if "_accum" in op:
            def operation(i):
                return torch.addmm(destinations[i],a,b,alpha=.5,beta=1.,out=destinations[i])
            def prepare(count):
                for destination in destinations[:count]: destination.copy_(inputs[2])
            operation.prepare = prepare
            return operation
        return lambda i: torch.mm(a,b,out=destinations[i])
    if op == "dot_general_axes":
        if case["variant"] == "batch":
            a, b = x.permute(1,0,2), inputs[1].permute(2,0,1)
            return lambda: torch.bmm(a,b)
        dims = (tuple(case["lhs_contracting_dims"]),tuple(case["rhs_contracting_dims"]))
        return lambda: torch.tensordot(x,inputs[1],dims=dims)
    if op == "dot_general_conj":
        lhs, rhs = case["lhs_conj"], case["rhs_conj"]
        if not lhs and not rhs:
            return lambda: torch.dot(x,inputs[1])
        if lhs and not rhs:
            return lambda: torch.vdot(x,inputs[1])
        if rhs and not lhs:
            return lambda: torch.vdot(inputs[1],x)
        raise ValueError("no single public both-conjugated dot counterpart")
    if op.startswith("dot_general_read_into"):
        count = 1 if verify_only else max(1, min(4096, (256 << 20) // (inputs[2].numel()*inputs[2].element_size())))
        destinations = [torch.empty_like(inputs[2]) for _ in range(count)]
        accumulating = "_into_accum" in op
        alpha, beta = (.5, 1.) if accumulating else (1., 0.)
        def operation(i):
            return torch.addmm(destinations[i], x, inputs[1], beta=beta, alpha=alpha, out=destinations[i])
        if accumulating:
            def prepare(count):
                for destination in destinations[:count]:
                    destination.copy_(inputs[2])
            operation.prepare = prepare
        return operation
    if op == "axpby_read_into_accum":
        count = 1 if verify_only else max(1, min(4096, (256 << 20) // (x.numel()*x.element_size())))
        destinations = [torch.empty_like(x) for _ in range(count)]
        alpha, beta = case.get("alpha", .5), case.get("beta", 1.)
        if beta == 1.:
            operation = lambda i: destinations[i].add_(x, alpha=alpha)
        elif beta == 0.:
            operation = lambda i: torch.mul(x, alpha, out=destinations[i])
        elif beta == 1.-alpha:
            operation = lambda i: torch.lerp(destinations[i], x, alpha, out=destinations[i])
        else:
            raise ValueError("no verified single-call PyTorch alpha/beta counterpart")
        def prepare(count):
            for destination in destinations[:count]:
                destination.copy_(inputs[1])
        operation.prepare = prepare
        return operation
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
    if op in {"scale_real", "scale_complex"}:
        return lambda: torch.mul(x, inputs[1])
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
        weight = inputs[1] if case.get("normalization_weight") else None
        bias = inputs[1 + int(case.get("normalization_weight", False))] if case.get("normalization_bias") else None
        return lambda: torch.nn.functional.layer_norm(x, (x.shape[1],), weight=weight, bias=bias, eps=1e-5)
    if op == "rms_norm":
        if case.get("normalization_bias"):
            raise NotImplementedError("PyTorch rms_norm has no bias argument")
        weight = inputs[1] if case.get("normalization_weight") else None
        return lambda: torch.nn.functional.rms_norm(x, (x.shape[1],), weight=weight, eps=1e-5)
    if op == "pad":
        padding = case.get("padding", [(0, 0, 0), (8, 8, 0)])
        if any(interior for _, _, interior in padding):
            raise NotImplementedError("PyTorch pad has no single-call interior padding counterpart")
        widths = tuple(v for lo, hi, _ in reversed(padding) for v in (lo, hi))
        return lambda: torch.nn.functional.pad(x, widths)
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
    if op == "broadcast_in_dim":
        shape = tuple(case["output_shape"])
        return lambda: torch.expand_copy(x, shape)
    if op == "convert":
        return lambda: x.to(dtype=torch.float64)
    if op == "gather":
        axis, indices = case["gather_axis"], inputs[1].reshape(-1)
        return lambda: torch.index_select(x, axis, indices)
    if op == "cast":
        dtype = torch.float64 if x.dtype == torch.float32 else torch.float32
        return lambda: x.to(dtype=dtype)
    if op in {"cholesky", "svdvals", "eigvalsh", "inv", "det", "norm"}:
        return lambda: getattr(torch.linalg, op)(x)
    if op == "pinv":
        return lambda: torch.linalg.pinv(x, atol=inputs[1], rtol=inputs[2])
    if op == "householder_r":
        packed, tau = torch.geqrf(x)
        argument = packed[:min(x.shape), :]
        return lambda: torch.triu(argument)
    if op == "householder_q_columns":
        packed, tau = torch.geqrf(x)
        start, end = case["column_range"]
        if start == 0 and end <= min(x.shape):
            packed, tau = packed[:, :end], tau[:end]
            return lambda: torch.linalg.householder_product(packed, tau)
        selector = torch.eye(x.shape[0], dtype=x.dtype, device=x.device)[:, start:end]
        packed = packed[:, :min(x.shape)]
        return lambda: torch.ormqr(packed, tau, selector, left=True, transpose=False)
    if op == "householder_qr":
        return lambda: torch.geqrf(x)
    if op == "svd_full":
        return lambda: torch.linalg.svd(x, full_matrices=True)
    if op == "slogdet":
        return lambda: torch.linalg.slogdet(x)
    if op == "triangular_solve":
        return lambda: torch.linalg.solve_triangular(x, inputs[1], upper=False)
    if op == "tensordot":
        return lambda: torch.tensordot(x, inputs[1], dims=1)
    if op == "vdot_read":
        return lambda: torch.vdot(x, inputs[1])
    if op == "norm_squared_read":
        return lambda: torch.vdot(x, x)
    if op == "reduce_sum_squares":
        return (lambda: torch.vdot(x, x)) if case.get("full_reduction") else (lambda: torch.linalg.vecdot(x, x, dim=1))
    if op.startswith("reduce_"):
        fn = {"reduce_sum": torch.sum, "reduce_prod": torch.prod, "reduce_max": torch.amax,
              "reduce_min": torch.amin, "reduce_mean": torch.mean}[op]
        return (lambda: fn(x)) if case.get("full_reduction") else (lambda: fn(x, dim=1))
    raise ValueError(f"missing native PyTorch API: {op}")


def jax_operation(jax, jnp, case):
    if case["dtype"] in {"i32", "i64"} and case["op"] == "div":
        return lambda x, y: jax.lax.div(x, y)
    if case["dtype"] in {"i32", "i64"} and case["op"] == "reduce_sum":
        return lambda x: jnp.sum(x, axis=1, dtype=x.dtype)
    if case["op"] in {"scale_real", "scale_complex"}:
        return lambda x, factor: jnp.multiply(x, factor)
    if case["op"].startswith("einsum_plan"):
        return lambda a,b: jnp.matmul(a,b)
    if case["op"] == "dot_general_axes":
        dims = ((tuple(case["lhs_contracting_dims"]),tuple(case["rhs_contracting_dims"])),
                (tuple(case["lhs_batch_dims"]),tuple(case["rhs_batch_dims"])))
        precision = jax.lax.Precision.HIGHEST
        return lambda a,b: jax.lax.dot_general(a,b,dims,precision=precision)
    if case["op"] == "dot_general_conj":
        lhs, rhs = case["lhs_conj"],case["rhs_conj"]
        if not lhs and not rhs:
            return lambda a,b: jnp.dot(a,b)
        if lhs and not rhs:
            return lambda a,b: jnp.vdot(a,b)
        if rhs and not lhs:
            return lambda a,b: jnp.vdot(b,a)
        raise ValueError("no single public both-conjugated dot counterpart")
    if case.get("read_op"):
        return jax_operation(jax, jnp, dict(case, op=case["read_op"], read_op=""))
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
        padding = tuple(tuple(v) for v in case.get("padding", [(0, 0, 0), (8, 8, 0)]))
        zero = jax.device_put(np.array(0, dtype={"f32": np.float32, "f64": np.float64}[case["dtype"]]), jax.devices("gpu")[0])
        jax.block_until_ready(zero)
        return lambda x: jax.lax.pad(x, zero, padding)
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
    if op == "broadcast_in_dim":
        shape, dims = tuple(case["output_shape"]), tuple(case["broadcast_dims"])
        return lambda x: jax.lax.broadcast_in_dim(x, shape, dims)
    if op == "convert":
        return lambda x: x.astype(jnp.float64)
    if op == "dynamic_slice":
        sizes = tuple(case["slice_sizes"])
        return lambda x, *starts: jax.lax.dynamic_slice(x, starts, sizes, allow_negative_indices=False)
    if op == "gather":
        axis = case["gather_axis"]
        dims = jax.lax.GatherDimensionNumbers(offset_dims=(1-axis,), collapsed_slice_dims=(axis,), start_index_map=(axis,))
        sizes = tuple(1 if i == axis else n for i, n in enumerate(case["shape"]))
        return lambda x, indices: jax.lax.gather(x, indices, dims, sizes, mode="promise_in_bounds")
    if op == "cast":
        dtype = jnp.float64 if case["dtype"] == "f32" else jnp.float32
        return lambda x: x.astype(dtype)
    if op in {"cholesky", "eigvalsh", "inv", "det", "norm"}:
        return lambda x: getattr(jnp.linalg, op)(x)
    if op == "svdvals":
        return lambda x: jnp.linalg.svd(x, compute_uv=False)
    if op == "pinv":
        return lambda x, atol, rtol: jnp.linalg.pinv(x, rtol=rtol)
    if op == "householder_r":
        return lambda packed: jnp.triu(packed)
    if op == "householder_q_columns":
        return lambda packed, tau: jax.lax.linalg.householder_product(packed, tau)
    if op == "householder_qr":
        return lambda x: jnp.linalg.qr(x, mode="raw")
    if op == "svd_full":
        return lambda x: jnp.linalg.svd(x, full_matrices=True)
    if op == "slogdet":
        return lambda x: jnp.linalg.slogdet(x)
    if op == "triangular_solve":
        import jax.scipy.linalg
        return lambda x, rhs: jax.scipy.linalg.solve_triangular(x, rhs, lower=True)
    if op == "tensordot":
        return lambda x, y: jnp.tensordot(x, y, axes=1)
    if op == "vdot_read":
        return lambda x, y: jnp.vdot(x, y)
    if op == "norm_squared_read":
        return lambda x: jnp.vdot(x, x)
    if op == "reduce_sum_squares":
        return (lambda x: jnp.vdot(x, x)) if case.get("full_reduction") else (lambda x: jnp.vecdot(x, x, axis=1))
    if op.startswith("reduce_"):
        fn = getattr(jnp, op.removeprefix("reduce_"))
        axis = None if case.get("full_reduction") else 1
        return lambda x: fn(x, axis=axis)
    raise ValueError(f"missing native JAX API: {op}")


def prepare_jax_inputs(case, inputs):
    """Adapt vector start indices to JAX's scalar-index API before any clock."""
    if case["op"].startswith("einsum_plan") and "strided" in case["op"]:
        return [inputs[0].T,inputs[1].T]
    if case.get("read_op"):
        return [value.T for value in inputs]
    if case["op"] in {"householder_r", "householder_q_columns"}:
        import jax.numpy as jnp
        h, tau = jnp.linalg.qr(inputs[0], mode="raw")
        packed = h.T
        if case["op"] == "householder_r":
            return [packed[:min(case["shape"]), :]]
        start, end = case["column_range"]
        if start != 0 or end > min(case["shape"]):
            raise ValueError("no matched JAX range-Q operation")
        return [packed[:, :end], tau[:end]]
    if case["op"] == "dynamic_slice":
        return [inputs[0], *(inputs[1][axis] for axis in range(len(case["slice_sizes"])))]
    return inputs
