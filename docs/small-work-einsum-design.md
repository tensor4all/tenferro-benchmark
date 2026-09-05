# Public F64 einsum small-work cases

The suite adds 18 square column-major `ij,jk->ik` cases at dimensions 2, 4
and 16 to the existing 24 add workflows. No full family/layout/dtype coverage
or performance acceptance is claimed by this matrix alone.

Each size has concrete-fresh, concrete-shared, eager-no-ad, eager-ad,
prepared-setup and prepared-repeat cases. Ordinary calls use the existing
`TensorEinsumExt` / `EagerEinsumExt`; preparation uses
`ConcreteEinsumPlan::prepare`, repeated execution uses `plan.execute`.
For this binary matmul shape, eager dispatch can use its direct dot-general
path: these cases do not measure generic extension fallback overhead.

The four ordinary tiers bind to the canonical concrete/eager einsum route IDs.
Setup binds to `einsum.einsum.prepare.concrete`, phase `setup`, and reports
provider `not-applicable`: plan construction is backend-independent. The
in-development contract v2 adds `prepared-setup` as a tier, not a public API.
Prepared execution binds to `einsum.einsum.prepared.concrete`, phase `execution`.
Wrong setup/repeat phases are rejected. Matrix size in child argv is element
count; producer descriptors independently report the square matrix shape.

Input/backend creation and correctness are outside timing. Fresh calls include
session entry/exit; shared calls enter once outside measurement. Setup includes
new plan creation and destruction. Prepared-repeat keeps plan and session outside
measurement but retains the public execute revalidation. Each invocation starts
from original inputs and drops its output; only add supports dependent-ten calls.

Dyadic nonsymmetric fixtures use an independent column-major triple-loop reference.
Checks compare exact shape, F64 dtype, finite values and every component. The AD
case differentiates the scalar sum of all outputs outside timing and checks both
full gradients against `dA[i,j] = sum_k B[j,k]` and `dB[j,k] = sum_i A[i,j]`.
The size-2 reference has a separate known-value assertion. Setup correctness also
executes a prepared plan; it is not inferred from a plan's shape alone.

Existing calibration, telemetry, process repetition, receipt verification, raw
archive and publication gates are reused. Correctness-only runs do not establish
release readiness or a valid baseline/candidate comparison. C64, layouts,
compiled-repeat and other family coverage remain required under #1758.
