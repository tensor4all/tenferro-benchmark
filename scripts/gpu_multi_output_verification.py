"""All-output, gauge-invariant checks for CUDA public factorizations."""
import numpy as np


def leaves(value):
    return list(value) if isinstance(value, (tuple, list)) else [value]


def verify_outputs(actual, expected, case, inputs, verify_one):
    actual, expected = leaves(actual), leaves(expected)
    if len(actual) != len(expected):
        raise ValueError("verification output arity mismatch")
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
