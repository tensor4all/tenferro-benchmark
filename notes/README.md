# Benchmark item notes

Persistent, hand-maintained analysis lives here, separate from generated results.
For `result/<target>/<suite>.md`, use `notes/<target>/<suite>.md` and an explicit
`<a id="problem_id"></a>` anchor for each documented benchmark item.

The standard GPU formatter (`scripts/format_gpu_results.py`) links documented
problem names to these anchors when writing under this repository's `result/`
directory. Missing notes or anchors leave the item unchanged; stdout and outputs
outside `result/` have no automatic links. Other formatters are not yet connected.
Regeneration updates measured results but never overwrites notes.

Each item should record:

- Applicable backend, hardware, source revision and measurement conditions.
- Established findings, separately from hypotheses and unverified scope.
- Candidate status, failed checks and remaining work; do not present a diagnostic
  candidate as released behavior.
- Links to small, version-controlled evidence and reproducible patches where
  useful. Large local traces/binaries may be named as local artifacts, but notes
  must remain understandable without them.

Update or explicitly supersede findings when the implementation changes. Do not
copy generated publication tables here; retain only analysis-specific evidence.

Current notes: [NVIDIA dense](nvidia-gpu/gpu/dense.md),
[NVIDIA elementwise](nvidia-gpu/gpu/elementwise.md).
