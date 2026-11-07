#!/usr/bin/env python3
"""The checks a linalg JVP/VJP row can be held to, and the contract around them.

A suite problem's ``verify.mode`` names the check that backs its row:

``cpu_reference`` (default, and the only mode that existed before)
    Re-execute the same loss through ``torch.func`` in f64 and compare every
    element of the primal and of the derivative.  This is the strongest check
    and the most expensive one: at n=4096 it costs a second AD pass over an
    SVD/eigh, which dwarfs the measurement it checks.

``analytic``
    Build the derivative in closed form -- ``d sum(svd(a))/da = U V^T`` for a
    full-rank square matrix and ``d tr(a)/da = I`` -- and compare every element,
    at the same ``rtol``/``atol``.  The reference is one primal factorisation or
    none at all, so the check costs a fraction of ``cpu_reference``.

``finite_difference``
    Compare the row's directional derivative against
    ``(L(A + tV) - L(A - tV)) / 2t``.  A central difference cannot be exact, so
    the step ``t`` is derived from f64 precision and the perturbation scale
    (``t = eps^(1/3) * ||A|| / ||V||``) rather than tuned per case, and the
    verdict still uses the manifest's ``rtol``/``atol`` -- nothing is loosened
    to make a row pass.

``measure_only``
    Run no check at all.  Refused unless the same ``verify`` block states a
    non-empty ``reason``: a measurement that carries no evidence has to say so
    in its own manifest, not in a commit message.

Every mode returns the same record fields (``status``, ``mode``,
``max_abs_error``, ``max_rel_error``, ``residual``, ``reason``), so a reader
learns which check backed a number without learning a second schema.
"""

from __future__ import annotations

import numpy as np

MODES = ("cpu_reference", "analytic", "finite_difference", "measure_only")
DEFAULT_MODE = "cpu_reference"

#: Losses with a closed form for dL/dA.
ANALYTIC_LOSSES = ("grad_sum_svd_s", "grad_sum_eigh")

#: Losses whose derivative has no reference here that is cheaper than the
#: directional finite difference.
FINITE_DIFFERENCE_LOSSES = ("grad_sum_qr", "grad_sum_lu", "grad_sum_solve")

_EPS = float(np.finfo(np.float64).eps)
#: Exponent of the classic central-difference step for a first derivative:
#: truncation error grows like t^2 while rounding error grows like eps/t.
FD_STEP_EXPONENT = 1.0 / 3.0


def mode_for(problem: dict, defaults: dict | None = None) -> str:
    """The verification mode of one problem, with the suite defaults filled in.

    Fails on an unknown mode and on ``measure_only`` without a non-empty
    ``reason``, mirroring ``src/verification_mode.rs`` so the manifest cannot
    pass one reader and fail the other.
    """
    block = (problem or {}).get("verify") or {}
    default_block = (defaults or {}).get("verify") or {}
    mode = block.get("mode", default_block.get("mode", DEFAULT_MODE))
    if mode not in MODES:
        raise ValueError(
            f"{problem.get('id', '<problem>')}: unknown verify.mode={mode!r}; "
            f"expected one of {', '.join(MODES)}"
        )
    if mode == "measure_only" and not _reason(problem, defaults):
        raise ValueError(
            f"{problem.get('id', '<problem>')}: verify.mode=measure_only requires a "
            "non-empty verify.reason; a measurement with no check must say why it "
            "has none"
        )
    return mode


def _reason(problem: dict, defaults: dict | None = None) -> str | None:
    block = (problem or {}).get("verify") or {}
    default_block = (defaults or {}).get("verify") or {}
    reason = block.get("reason", default_block.get("reason"))
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    return None


def measure_only_reason(problem: dict, defaults: dict | None = None) -> str | None:
    """The recorded reason a ``measure_only`` problem carries no check."""
    return _reason(problem, defaults) if mode_for(problem, defaults) == "measure_only" else None


# ---------------------------------------------------------------------------
# torch plumbing
# ---------------------------------------------------------------------------


def resolve_device(preferred: str | None = None) -> str:
    """The device a reference is built on: the accelerator when one is present.

    A closed-form reference built on the same GPU as the row costs a fraction of
    the CPU re-execution it replaces, which is the whole point of the cheaper
    modes; f64 is required on both sides.
    """
    import torch

    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def _loss_fn(name: str):
    from benchmark_gpu_linalg_ad import LOSS_TORCH

    try:
        return LOSS_TORCH[name]
    except KeyError as exc:  # pragma: no cover - guarded by the caller
        raise ValueError(f"no reference loss for {name!r}") from exc


def _tensors(matrix, tangent, rhs, device):
    import torch

    torch.set_num_threads(1)
    x = torch.as_tensor(np.asarray(matrix), dtype=torch.float64, device=device)
    t = torch.as_tensor(np.asarray(tangent), dtype=torch.float64, device=device)
    b = None if rhs is None else torch.as_tensor(np.asarray(rhs), dtype=torch.float64, device=device)
    return x, t, b


def _as_numpy(values):
    return [np.asarray(value.detach().cpu().numpy().copy()) for value in values]


# ---------------------------------------------------------------------------
# Closed-form reference
# ---------------------------------------------------------------------------


def analytic_reference(
    loss: str,
    phase: str,
    matrix,
    tangent,
    rhs=None,
    *,
    device: str | None = None,
):
    """The primal and derivative of ``loss`` at ``matrix``, in closed form.

    ``grad_sum_svd_s``: for a full-rank square matrix, ``d sum(s) / dA = U V^T``
    with ``A = U diag(s) V^T``.  ``grad_sum_eigh``: ``L = tr(A)`` for an SPD
    matrix, so ``dL/dA = I``.  Both are compared against the row element by
    element, exactly like ``cpu_reference``; only the reference changes.
    """
    import torch

    if loss not in ANALYTIC_LOSSES:
        raise ValueError(
            f"{loss}: no closed-form derivative here; verify it with "
            "verify.mode=finite_difference"
        )
    if phase not in ("jvp", "vjp"):
        raise ValueError(f"unsupported phase: {phase}")
    device = resolve_device(device)
    x, t, _ = _tensors(matrix, tangent, rhs, device)
    with torch.no_grad():
        if loss == "grad_sum_svd_s":
            u, s, vh = torch.linalg.svd(x, full_matrices=False)
            gradient = u @ vh
            primal = s.sum()
        else:
            gradient = torch.eye(x.shape[0], dtype=torch.float64, device=device)
            primal = torch.diagonal(x).sum()
        derivative = (gradient * t).sum() if phase == "jvp" else gradient
    return _as_numpy([primal, derivative]), {
        "reference": (
            "closed form dL/dA = U V^T from the accelerator's own SVD factors"
            if loss == "grad_sum_svd_s"
            else "closed form dL/dA = I for L = tr(A) on an SPD matrix"
        ),
        "device": device,
    }


# ---------------------------------------------------------------------------
# Central difference of the primal loss
# ---------------------------------------------------------------------------


def fd_step(loss: str, matrix, tangent, rhs=None, *, device: str | None = None) -> float:
    """The perturbation ``t`` in ``A +- t V``.

    ``eps**(1/3) * ||A||_F / ||V||_F`` is the classic balance of a central
    difference's ``t^2`` truncation error against its ``eps/t`` rounding error,
    so the perturbation is a ``eps**(1/3)`` relative move of the input whatever
    the matrix size.  It is derived, not fitted per case.
    """
    import torch

    device = resolve_device(device)
    x, t, _ = _tensors(matrix, tangent, rhs, device)
    with torch.no_grad():
        scale = float(torch.linalg.matrix_norm(x)) / max(float(torch.linalg.matrix_norm(t)), 1e-300)
    return _EPS**FD_STEP_EXPONENT * scale


def finite_difference_directional(
    loss: str,
    phase: str,
    matrix,
    tangent,
    rhs=None,
    *,
    device: str | None = None,
    step: float | None = None,
):
    """``(primal, dL/dA . V)`` for one loss, from three primal evaluations."""
    import torch

    if phase not in ("jvp", "vjp"):
        raise ValueError(f"unsupported phase: {phase}")
    device = resolve_device(device)
    x, t, b = _tensors(matrix, tangent, rhs, device)
    base = _loss_fn(loss)
    fn = base if b is None else (lambda a: base(a, b))
    step = fd_step(loss, matrix, tangent, rhs, device=device) if step is None else step
    with torch.no_grad():
        primal = float(fn(x))
        forward = float(fn(x + step * t))
        backward = float(fn(x - step * t))
    directional = (forward - backward) / (2 * step)
    return primal, directional, {
        "step": step,
        "primal": primal,
        "forward": forward,
        "backward": backward,
        "directional": directional,
        "step_rule": "eps**(1/3) * ||A||_F / ||V||_F",
        "device": device,
    }


def directional_of_actual(phase: str, derivative, tangent) -> float:
    """The directional derivative a row's derivative output states along ``V``.

    A JVP row reports ``dL/dA . V`` as a scalar; a VJP row reports the full
    gradient, so its directional derivative is ``<G, V>``.  Both are one number
    that a central difference can be compared with.
    """
    derivative = np.asarray(derivative)
    if phase == "jvp":
        if derivative.size != 1:
            raise ValueError("a jvp row must report a scalar directional derivative")
        return float(derivative.reshape(()))
    return float(np.sum(derivative * np.asarray(tangent)))


# ---------------------------------------------------------------------------
# One verdict, whichever check produced it
# ---------------------------------------------------------------------------


def _worst(actual, reference, rtol: float, atol: float):
    """``(max_abs, max_rel, worst_over_tolerance)`` for one pair of outputs."""
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    if actual.shape != reference.shape:
        raise ValueError(f"AD output shape mismatch: {actual.shape} vs {reference.shape}")
    if actual.dtype != np.dtype("float64"):
        raise ValueError(f"AD output dtype mismatch: {actual.dtype}")
    if not np.all(np.isfinite(actual)):
        raise ValueError("nonfinite AD output")
    if not np.all(np.isfinite(reference)):
        raise ValueError("nonfinite reference")
    error = np.abs(actual - reference)
    max_abs = float(np.max(error)) if error.size else 0.0
    max_rel = float(np.max(error / np.maximum(np.abs(reference), atol))) if error.size else 0.0
    worst = float(np.max(error / (atol + rtol * np.abs(reference)))) if error.size else 0.0
    return max_abs, max_rel, worst


def check(
    mode: str,
    loss: str,
    phase: str,
    actual,
    matrix,
    tangent,
    rhs,
    rtol: float,
    atol: float,
    *,
    device: str | None = None,
    step: float | None = None,
) -> dict:
    """Hold one row's outputs to ``mode`` and return the record's verification fields.

    ``cpu_reference`` and ``analytic`` compare every element of the primal and
    the derivative.  ``finite_difference`` compares the primal element by
    element and the directional derivative as one number.  The verdict always
    uses the manifest's own ``rtol``/``atol``: a check that cannot meet them
    fails instead of loosening them.
    """
    actual = [np.asarray(value) for value in actual]
    if not actual:
        raise ValueError("row has no outputs to check")
    if mode == "cpu_reference":
        from gpu_linalg_ad_contracts import cpu_reference

        reference, diagnostic = cpu_reference(
            _loss_fn(loss), matrix, tangent, rhs, phase
        )
        pairs = list(zip(actual, reference))
        backend = "cpu_fp64 AD re-execution"
        reason = None
    elif mode == "analytic":
        reference, diagnostic = analytic_reference(
            loss, phase, matrix, tangent, rhs, device=device
        )
        pairs = list(zip(actual, reference))
        backend = f"{diagnostic['reference']} ({diagnostic['device']})"
        reason = None
    elif mode == "finite_difference":
        primal, directional, diagnostic = finite_difference_directional(
            loss, phase, matrix, tangent, rhs, device=device, step=step
        )
        if len(actual) < 2:
            raise ValueError("a finite_difference row needs a primal and a derivative")
        measured = directional_of_actual(phase, actual[1], tangent)
        pairs = [
            (actual[0], np.asarray(primal)),
            (np.asarray(measured), np.asarray(directional)),
        ]
        backend = (
            f"central difference of the primal loss on {diagnostic['device']} "
            f"(t={diagnostic['step']:.3e} = {diagnostic['step_rule']})"
        )
        reason = (
            f"finite_difference: t={diagnostic['step']:.6e}, "
            f"directional={directional:.12e}, measured={measured:.12e}, "
            f"error={abs(measured - directional):.3e}"
        )
    else:
        raise ValueError(f"{mode}: no check to run")

    max_abs = max_rel = worst = 0.0
    for observed, expected in pairs:
        abs_error, rel_error, ratio = _worst(observed, expected, rtol, atol)
        max_abs = max(max_abs, abs_error)
        max_rel = max(max_rel, rel_error)
        worst = max(worst, ratio)
    verdict = {
        "mode": mode,
        "status": "passed" if worst <= 1.0 else "failed",
        "reference_backend": backend,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "residual": worst,
        "rtol": rtol,
        "atol": atol,
        "reason": reason,
        "diagnostic": diagnostic,
    }
    if verdict["status"] != "passed":
        raise VerificationFailure(
            f"{mode} check failed: error/tolerance={worst:.3e} "
            f"(max_abs={max_abs:.3e})",
            verdict,
        )
    return verdict


class VerificationFailure(ValueError):
    """A row whose output does not meet the check it declared."""

    def __init__(self, message: str, verdict: dict):
        super().__init__(message)
        self.verdict = verdict


def not_run(problem: dict, defaults: dict | None = None) -> dict:
    """The verification block of a row that ran no check, and why."""
    reason = _reason(problem, defaults)
    if not reason:
        raise ValueError("measure_only record without a reason")
    return {
        "mode": "measure_only",
        "status": "not_run",
        "reference_backend": None,
        "max_abs_error": None,
        "max_rel_error": None,
        "residual": None,
        "rtol": None,
        "atol": None,
        "reason": reason,
    }
