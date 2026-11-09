"""All-output, gauge-invariant checks for CUDA public factorizations."""
import numpy as np


def leaves(value):
    return list(value) if isinstance(value, (tuple, list)) else [value]


def budget(case, reference, exact=None):
    """The per-element allowance, refused when the comparison cannot express it.

    A budget that is not finite scores every miss as 0: `rtol = inf` makes every
    threshold infinite, and `atol + rtol * abs(1.7e308+1.7e308j)` overflows to the
    same from finite components. `exact` masks the elements the comparison already
    found equal, which get a budget of exactly 1.
    """
    rtol, atol = case["rtol"], case["atol"]
    if not all(np.isfinite(tolerance) and tolerance >= 0 for tolerance in (rtol, atol)):
        raise ValueError(f"verification tolerance must be finite and non-negative: rtol={rtol!r} atol={atol!r}")
    with np.errstate(over="ignore", invalid="ignore"):
        scale = atol + rtol * np.abs(reference)
    if exact is not None:
        scale = np.where(exact, 1, scale)
    if not np.all(np.isfinite(scale)):
        raise ValueError(f"verification tolerance budget is not finite: rtol={rtol!r} atol={atol!r}")
    return scale


def scored(values, allowance):
    """|miss| / allowance, with 0/0 = 0 and a miss outside a zero allowance = inf.

    A plain division produced NaN wherever the reference element and the miss
    were both exactly zero. With atol = 0 every zero of the identity reference
    carries an allowance of 0, so a scaled u whose u.T @ u = 8 I scored NaN off
    the diagonal, np.max returned NaN, and max(worst, NaN) dropped the whole
    comparison. Checking finiteness after the reduction cannot recover it; the
    comparison has to be scored here.
    """
    miss = np.abs(values)
    out = np.zeros(miss.shape, dtype=np.float64)
    positive = allowance > 0
    out[positive] = miss[positive] / allowance[positive]
    out[~positive] = np.inf
    out[~positive & (miss == 0)] = 0.0
    return out


def verify_outputs(actual, expected, case, inputs, verify_one):
    actual, expected = leaves(actual), leaves(expected)
    if len(actual) != len(expected):
        raise ValueError("verification output arity mismatch")
    if case["op"] == "householder_qr":
        for a, e in zip(actual, expected):
            if a.shape != e.shape or a.dtype != e.dtype or not np.all(np.isfinite(a)):
                raise ValueError("verification compact-QR shape/dtype/finiteness mismatch")
        packed, tau = actual
        dtype = np.complex128 if np.iscomplexobj(packed) else np.float64
        packed, tau = packed.astype(dtype), tau.astype(dtype)
        m, n = inputs[0].shape
        k = min(m, n)
        q = np.eye(m, dtype=dtype)
        for j in range(k):
            v = np.concatenate((np.ones(1, dtype=dtype), packed[j+1:, j]))
            q[:, j:] -= tau[j]*np.outer(q[:, j:] @ v, v.conj())
        r = np.triu(packed[:k, :])
        a = inputs[0].astype(dtype)
        worst = float(np.max(scored(q[:, :k] @ r-a, budget(case, a))))
        identity = np.eye(m)
        worst = max(worst, float(np.max(scored(q.conj().T @ q-identity, budget(case, identity)))))
        if not np.isfinite(worst) or worst > 1:
            raise ValueError(f"verification compact-QR error/tolerance = {worst}")
        return worst
    if case["op"] != "svd_full":
        return max(verify_one(a, e, case) for a, e in zip(actual, expected))
    for a, e in zip(actual, expected):
        if a.shape != e.shape or a.dtype != e.dtype or not np.all(np.isfinite(a)):
            raise ValueError("verification full-SVD shape/dtype/finiteness mismatch")
    u, s, vh = actual
    dtype = np.complex128 if np.iscomplexobj(u) else np.float64
    u, vh, a = u.astype(dtype), vh.astype(dtype), inputs[0].astype(dtype)
    m, n = a.shape
    k = min(m, n)
    if np.any(s < 0) or np.any(s[1:] > s[:-1]):
        raise ValueError("verification singular values are not sorted/nonnegative")
    reconstructed = (u[:, :k] * s.astype(np.float64)) @ vh[:k, :]
    worst = float(np.max(scored(reconstructed-a, budget(case, a))))
    for gram in (u.conj().T @ u, vh @ vh.conj().T):
        identity = np.eye(len(gram))
        worst = max(worst, float(np.max(scored(gram-identity, budget(case, identity)))))
    worst = max(worst, verify_one(s, expected[1], case))
    if not np.isfinite(worst) or worst > 1:
        raise ValueError(f"verification full-SVD error/tolerance = {worst}")
    return worst


def normalized_outputs(actual, case, backend):
    if case["op"] == "dot_general_axes" and case.get("variant") == "batch" and backend in {"pytorch-cuda","jax-cuda","jax-cuda-jit"}:
        return [leaves(actual)[0].transpose(1,2,0)]
    values = leaves(actual)
    if case["op"] == "householder_qr" and backend.startswith("jax"):
        # JAX raw QR stores native reflector rows; inspect the logical packed
        # columns only after the operation and its clock have completed.
        values[0] = values[0].T
    return values
