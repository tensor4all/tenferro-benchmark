# GPU collection triage

## What actually failed

The diagnostic artifacts for all three 2026-10-31 scheduled runs contain the
same failure set, not missing references or SKU exclusions:

| Run | Slot UTC | failed case | missing / duplicates / unexpected |
|---|---|---|---|
| [41183315922](https://github.com/tensor4all/tenferro-benchmark/actions/runs/41183315922) | 08 | `gpu/tensornetwork:tensornetwork_permutation_optimized_f32:tenferro-cuda-trace` | 0 / 0 / 0 |
| [41208240314](https://github.com/tensor4all/tenferro-benchmark/actions/runs/41208240314) | 14 | same singleton | 0 / 0 / 0 |
| [41237278524](https://github.com/tensor4all/tenferro-benchmark/actions/runs/41237278524) | 20 | same singleton | 0 / 0 / 0 |

Each has 40 executed pairs, one failed pair, and no unsupported pairs. The
failed row has `status: runtime_failed`, with this execution reason:

```text
run: TypedTensor::host_data: runtime state failure: backend storage cannot be borrowed as host data; download explicitly first
```

This is an implementation/runtime failure, **not** evidence of an absent
reference, unsupported library operation, or unsuitable GPU SKU. Its
`verification.status: skipped` means verification could not run after the
execution exception; it does not make the case benign. The failure occurred
inside `Runtime::run`, before output download/verification. The exact internal
call site has not been reproduced or established; this change does not claim
to fix the runtime exception. The remaining execution defect is tracked in
[issue #291](https://github.com/tensor4all/tenferro-benchmark/issues/291).

All three used harness revision `5220eed2525b73a328ef426226d299c741c41aa8`,
but the library revisions and devices differed:

| Run | tenferro-rs commit | Device |
|---|---|---|
| 41183315922 | `e04e77d63589fbada427f3b472d34a432153b5bc` | A100 80GB PCIe |
| 41208240314 | `2f7aedafda5fde35daea1ca6f24b3432df0e1bc0` | A100 80GB PCIe |
| 41237278524 | `c15dfcf0ea5252effdef24cffbebf2f445916e20` | A100-SXM4-80GB |

Selection, case accounting, records and run metadata were read from each
diagnostic ZIP; large sparse-cache `.npz` members were not needed for this
comparison. The same exception occurred on both A100 variants; nothing in
these records establishes a benign SKU exclusion.

## Why the October 27 success is not a regression baseline

[40428820583](https://github.com/tensor4all/tenferro-benchmark/actions/runs/40428820583)
was a manual `perf-issues-readback` run, not the routine GPU matrix. Its log
and `runpod-gpu-readback-40428820583-attempt1` artifact contain the readback
cases under `gpu/perf_issues`, not the failed tensor-network trace case.
Consequently, that success cannot establish when this failure began.
The tracked `gpu/tensornetwork/20261006_193619/records.jsonl` does contain a
numerically passed trace row for this problem on an older implementation.
The failure is therefore not timeless, but its introduction has not been
bisected between those different revisions/routes.

## Classification and retry decision

Explicit `skipped` and legacy `not_configured` rows are counted as skipped;
explicit `unsupported` rows remain unsupported. Both require a nonblank reason
in the existing result fields, retained in the accounting's `reasons` map.
Reasonless exemptions and contradictory numerical failures are failed.
Runtime errors, exceptions, timeouts and mismatches are never classified by
matching benign-looking words in their error messages.

Failed, missing, duplicate and unexpected pairs keep a nonzero collector exit
code. Skipped/unsupported-only collections exit zero, but do not become
complete *measured* coverage. Counts and IDs/reasons are printed, and
`case_status.json` is written before a failing exit. Quick-from-full
accounting only omits known full-only IDs: unknown IDs and duplicates of
selected pairs still fail.

No `weekly_gate.py` or schedule change is appropriate for the observed
failure: it is not known-benign. A real runtime failure must not be recorded
as that week's success to avoid a retry. The existing success gate already
suppresses later slots after a genuinely successful (including benign-only)
collection. No retry slots were added, removed or dispatched. Workflow upload
and ingestion changes belong to the separate ingestion task and are not part
of this change.

## Verification boundary

The historical record/selection sets replay as the same singleton failure for
all three runs, with the diagnostic file retained on each failing CLI exit.
Focused tests exercise reasoned skipped/unsupported success, reasonless
exemption rejection, exceptions, timeout records, numerical failures, missing,
duplicate and unexpected records, including restricted accounting. A stubbed
end-to-end collector check also verifies that verification-only execution
failures and reasonless exemptions retain accounting JSON. Schema-only errors
remain nonzero even when their status would otherwise be a benign skip.

On primerose, existing tools were used without installing a CUDA toolchain:
NVIDIA A100 80GB PCIe, driver 580.178.04; installed nvcc 12.6.85; the existing
PyTorch 2.12.0+cu130 environment uses CUDA runtime 13.0. Three 2x2/3x3/4x4
float64 matmuls passed the existing Python GPU runner's numerical checks in
`GPU_BENCH_VERIFY_ONLY=1` mode. PyTorch intra-op and inter-op counts were both
set and read back as one. The records were accounted as verification-only,
with no timing samples or publication. This is not a Rust trace reproduction
and does not validate the failing tensor-network route.

No measurement host or RunPod instance was used, no GPU was rented, and no
full benchmark was collected. There is **no performance claim**. With the
same library and runtime behavior, the next weekly routine run will still
fail on the real trace exception; classification alone must not turn it green.
