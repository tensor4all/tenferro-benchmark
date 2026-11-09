# Verification modes

Every published row states which check backs its number. The suite problem's
`verify.mode` chooses it, and the choice travels through the record, the
summary sidecar and the dashboard:

| `verify.mode` | What runs | Cost at n=4096 (f64, CPU) |
|---|---|---|
| `cpu_reference` (default) | the whole case re-executed through `torch.func` in f64, every element compared | a second AD pass plus two primal evaluations |
| `analytic` | every element compared against a closed-form derivative built on the accelerator | one primal factorisation |
| `finite_difference` | the directional derivative compared with a central difference of the primal loss | three primal evaluations (the loss at the point and at both ends of the difference) |
| `measure_only` | nothing; refused unless `verify.reason` is non-empty | none |

The verdict of the three checking modes is the same three numbers in the same
record fields (`status`, `mode`, `reference_backend`, `max_abs_error`,
`max_rel_error`, `residual`, `rtol`, `atol`, `reason`), so a reader learns which
check produced a number without learning a second schema. Every mode is held to
the suite's own `rtol`/`atol`; no mode loosens them.

The reference is built **once per row**. A row is downloaded twice (the initial
and the final execution), and both downloads are compared against the same
reference, because the reference depends on the case rather than on which
download it judges. `residual` is always a finite number: an out-of-tolerance
element reports its own `error/tolerance` ratio, never an infinity that no JSON
reader accepts, and `max_abs_error`/`max_rel_error` are clamped to the largest
finite double for the same reason. A `0` tolerance is an exact-equality demand:
an error of `1e-310` against a zero reference fails, it is not scored against a
`tiny` denominator.

## Which loss can use which mode

| loss | derivative | mode |
|---|---|---|
| `grad_sum_svd_s` | `U V^T` for a full-rank square matrix | `analytic` |
| `grad_sum_eigh` | `I` for `L = tr(A)` on an SPD matrix | `analytic` |
| `grad_sum_qr`, `grad_sum_lu`, `grad_sum_solve` | no cheaper reference than the directional derivative | `finite_difference` |

`analytic` builds its reference from the row's own matrix, so a wrong SVD or
eigh convention is caught: the closed form and the CPU AD reference agree
element by element at the manifest tolerance, which is what
`tests/test_linalg_ad_verify_modes.py` holds them to at n=32..128.

## The finite-difference step

The central difference uses

```
t = eps^(1/3) * ||A||_F / ||V||_F
```

which is the classic balance of a central difference's `t^2` truncation error
against its `eps/t` rounding error. It is derived from `f64` precision and the
perturbation, not fitted per case: at n=32 the step is 2.8e-06 and at n=128 it
stays the same relative perturbation of the input.

The test sweeps `t`, `t/10` and `10t` and requires the derived step to be the
best of the three for the losses whose second derivative is non-zero (smaller is
rounding-limited, larger is truncation-limited), and requires every loss to
land under 1/100 of the tolerance budget at the derived step.

## `measure_only`

`measure_only` exists as a last resort for a case no reference can reach. It is
refused when `reason` is missing or blank, at manifest load and again in the
record schema, so a measurement cannot publish itself as verified:

* `verification.status` is `not_run`, never `passed`;
* `verification.mode` is required: `not_run` alone does not say who did not
  check, and a row without it could still carry a `reference_backend` and a
  `residual`, which reads as evidence;
* `status` stays `ok`, because the measurement did run; a case that never ran
  says `not_run` when its problem is `measure_only` and `skipped` for every
  other mode;
* `verification.reason` says why no check ran;
* the outputs are never copied to the host, because the mode compares nothing;
* the dashboard badges the row `unverified` (dashed, distinct from `fail`) with
  the reason in the title.

Records folded from a family runner carry only the `reference_backend` that the
row recorded. Nothing fills in `cpu_fp64`: a row that checked against a
different reference, or against none, must not read as a CPU-reference row.

No current manifest uses it: n=4096 cases are verified with `analytic` and
`finite_difference` instead.
