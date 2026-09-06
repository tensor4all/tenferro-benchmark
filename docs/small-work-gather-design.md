# Small-work gather slice

Six cases use the existing public `TensorIndexing::gather` on a BackendSession,
not a nonexistent Tensor::gather or an eager alias. Bind to
`core.gather.ordinary.concrete`. Data/output are F64; index tensors are I64.
Sizes4/16/256 use non-monotonic, repeated, in-range indices. A host indexed lookup
provides the independent full-value oracle. Check that data and indices remain
unchanged and that the output shape/dtype match.

Fresh/shared session boundaries reuse the existing concrete runner. As in the
existing public_api gather workload, GatherConfig is constructed per call.
`index_config_construction` is explicitly inside timing, alongside gather and
output lifetime; payload/index tensor construction and correctness stay outside.
This measures that declared ordinary workflow, not isolated kernel latency or
library-only overhead. Configuration construction is not silently attributed to
library validation. No new dependency, public API, cache or operation registry.

This slice adds an indexing representative; it does not establish eager/compiled,
other dtype/layout/invalid-input coverage or performance acceptance.
