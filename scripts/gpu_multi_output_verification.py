"""All-output, gauge-invariant checks for CUDA public factorizations."""
import numpy as np


def leaves(value):
    return list(value) if isinstance(value, (tuple, list)) else [value]


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
        worst = float(np.max(np.abs(q[:, :k] @ r-a)/(case["atol"]+case["rtol"]*np.abs(a))))
        identity = np.eye(m)
        worst = max(worst, float(np.max(np.abs(q.conj().T @ q-identity)/(case["atol"]+case["rtol"]*identity))))
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
    budget = case["atol"] + case["rtol"] * np.abs(a)
    worst = float(np.max(np.abs(reconstructed-a)/budget))
    for gram in (u.conj().T @ u, vh @ vh.conj().T):
        identity = np.eye(len(gram))
        worst = max(worst, float(np.max(np.abs(gram-identity)/(case["atol"]+case["rtol"]*identity))))
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
