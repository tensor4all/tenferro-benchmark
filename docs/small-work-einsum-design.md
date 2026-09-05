# Public F64 einsum small-work cases

## Compiled-repeat slice

The n2/4/16 F64 column-major `compiled-repeat` cases bind to
`einsum.einsum.prepared.traced`. Trace two input slots with `TraceContext`, use
ordinary `TraceContextEinsumExt::einsum`, compile with default `GraphCompiler`,
and execute through `Runtime::run_compiled`. No preset contraction tree, literal
input tensors, alternate compiler configuration or new library API is needed.
Trace/compile/runtime construction and input binding-array construction are
outside the repeat timer; runtime admission, execution and output lifetime are
inside. Correctness must run original, swapped, then original inputs on the same
program against independent matmul references. This covers execution, not the
still-required separate compilation/setup-cost measurement or C64 variants.

The suite includes 18 owned-input, 18 borrowed-input and three compiled-repeat
`ij,jk->ik` cases at dimensions 2, 4 and 16 alongside the existing 24 add workflows
(63 total). No full family/layout/dtype coverage
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
release readiness or a valid baseline/candidate comparison. C64, broader layouts,
compilation/setup-cost measurements and other family coverage remain required
under #1758; borrowed F64 layout coverage below does not establish complete layout
coverage.

## Borrowed F64 layouts

Owned tensors are compact column-major by contract (`tenferro-tensor/src/types.rs`);
changing their descriptor to row-major would not exercise that layout. These cases use the
existing `TypedTensorView::from_slice` and `TypedTensorReadEinsumExt::einsum_read`
instead, with separately identified `borrowed-fresh` and `borrowed-shared` cases.
These remain concrete execution of the canonical einsum operation, but are not
aliases for the measured owned-input cases. Preserve distinct case IDs, API tiers
and actual layout metadata; do not add a library API or another operation registry.

For each n=2/4/16, both session tiers exercise column-major control strides
`[1,n]`, row-major `[n,1]`, and padded non-contiguous `[2,2*n+1]` with offset 1.
Fill physical storage from the same logical nonsymmetric values used by the
existing independent triple-loop oracle. Poison padding with NaNs to expose
incorrect contiguous access. Construct storage and borrowed views outside timing;
include the complete public read call and any internal materialization in timing.
Fresh/shared session boundaries stay identical to their owned counterparts.
Only single-call workflows belong to this slice; do not imply borrowed eager,
AD or prepared support by materializing input before the public call.

The existing producer accepts `--layout`; non-column-major layouts are rejected
for owned-input tiers. The contract/schema and suite distinguish the borrowed
tiers without a second producer. Rust tests verify real view strides/offset,
every output element and both session modes. All 60 suite cases ran through the
actual producer with passed correctness and matching descriptors. After removing
unneeded stride-vector allocations, the affected 18 cases were rerun successfully;
the unchanged 42 owned/add cases retain the earlier execution evidence. None of
these correctness-only runs is performance acceptance. Timing still requires
release preparation, eligible resources and a frozen baseline/candidate campaign.
