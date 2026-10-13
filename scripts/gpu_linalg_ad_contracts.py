"""Independent CPU checks for matched scalar-loss CUDA JVP/VJP outputs."""
import numpy as np


def leaves(value):
    if isinstance(value, (tuple, list)):
        return [leaf for child in value for leaf in leaves(child)]
    return [value]


def cpu_reference(loss_fn, matrix, tangent, rhs, phase):
    import torch
    torch.set_num_threads(1)
    x = torch.as_tensor(matrix, dtype=torch.float64)
    t = torch.as_tensor(tangent, dtype=torch.float64)
    b = None if rhs is None else torch.as_tensor(rhs, dtype=torch.float64)
    fn = loss_fn if b is None else lambda a: loss_fn(a, b)
    if phase == "jvp":
        result = torch.func.jvp(fn, (x,), (t,))
        directional = float(result[1])
    else:
        primal, pullback = torch.func.vjp(fn, x)
        result = primal, pullback(torch.tensor(1., dtype=torch.float64))
        directional = float((result[1][0] * t).sum())
    # Check the derivative against a central difference of the primal loss.
    # This is setup/verification work, never part of a CUDA timed operation.
    step = 1e-5 / max(1., float(torch.linalg.vector_norm(t) / torch.linalg.vector_norm(x)))
    finite_difference = float((fn(x + step*t) - fn(x - step*t)) / (2*step))
    budget = 1e-4 * max(1., abs(finite_difference), abs(directional))
    if not np.isfinite(directional) or not np.isfinite(finite_difference) or abs(directional-finite_difference) > budget:
        raise ValueError(f"CPU AD/central-difference mismatch: {directional} vs {finite_difference}")
    expected = [v.detach().numpy().copy() for v in leaves(result)]
    return expected, dict(step=step, derivative=directional, finite_difference=finite_difference,
                         max_error_over_budget=abs(directional-finite_difference)/budget)


def verify_arrays(actual, expected, rtol, atol):
    if len(actual) != len(expected):
        raise ValueError("AD output arity mismatch")
    max_abs, max_rel, worst = 0., 0., 0.
    for value, reference in zip(actual, expected):
        value = np.asarray(value)
        if value.shape != reference.shape or value.dtype != np.dtype("float64"):
            raise ValueError("AD output shape/dtype mismatch")
        if not np.all(np.isfinite(value)) or not np.all(np.isfinite(reference)):
            raise ValueError("nonfinite AD output/reference")
        error = np.abs(value-reference)
        max_abs = max(max_abs, float(np.max(error)))
        max_rel = max(max_rel, float(np.max(error/np.maximum(np.abs(reference), atol))))
        worst = max(worst, float(np.max(error/(atol+rtol*np.abs(reference)))))
    if worst > 1:
        raise ValueError(f"AD error/tolerance={worst}")
    return max_abs, max_rel, worst
