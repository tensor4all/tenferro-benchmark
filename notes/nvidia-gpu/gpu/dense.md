# NVIDIA GPU dense: item notes

Analysis date: 2026-09-19. These findings describe the revisions below, not an
unqualified promise about the latest library or the current generated results.

<a id="dense_batched_matmul_f64_b1024_256"></a>
## Batched matmul, f64, batch=1024, m=n=k=256

### Scope and established findings

- A100 80GB PCIe; benchmark `9a5b104`; tenferro `40e24634` and the separately
  identified materialization-repair candidate; CubeCL `5939d8e3b1c479ed6db31de5f01cb6d432178f2d`.
- Eager and trace use the same
  `cutlass_80_tensorop_d884gemm_128x64_16x3_nn_align1` kernel as the investigated
  Torch run. GPU computation is about 2.06 ms. The additional wall time is mainly
  host-side management, not a slower matrix multiplication kernel.
- Earlier instrumentation found ~173 us in eager's first input-pointer request
  and ~113 us in trace's post-return flush. They are different observation points,
  not additive components proven to occur on every call.
- The historical CubeCL baseline's idle device-service thread polls with a 150 us sleep after its
  spin/yield budgets. OSRT traces observed ~200 us actual sleeps. Removing only
  that sleep improved large BMM in every paired round, supporting causality.
  This is per batched operation/service interaction, **not one wait per matrix**;
  a strict single-wait-per-call count has not been established.

### Historical candidate and verification status

Busy polling was rejected: CPU usage increased and small-case checks failed.
A second candidate retains batching and the spin/yield budgets, then uses
`park/unpark` notification when enqueue or flush publishes a complete task buffer.
It preserves task ownership, atomic publication ordering and completion waits.

- Large BMM reaches about 2.12–2.14 ms, without increased whole-process CPU-seconds.
- All 44 CubeCL-common unit tests pass, including startup and wakeup-race tests.
  All 60 numerical records in each of the two complete experiments pass.
- The notification candidate nevertheless **failed the original nonregression
  gate**: small eager batch=1 in round 2 regressed 40.8%, and eager chain 1k in
  round 3 regressed 20.2%. No production dependency pin was changed at that stage.

### Why the small cases varied

Subsequent diagnostics reproduce ~48/~70 us small-BMM modes in both binaries.
The small GPU kernel remains ~6.9 us in all four profiles. On this EPYC 7713P,
64 cores share one NUMA node but have eight separate L3 domains. Slow sampled
runs place the caller and `DSD-0-0` in different L3 domains; fast runs share one.

Restricting **both** binaries to CPUs 0–7 (one L3 domain), while retaining a 1T
provider, narrows small BMM to ~46–53 us. Four matched-placement pairs have
candidate/baseline ratios 0.863–1.030 for BMM and 0.971–1.023 for chain 1k.
This strongly supports CPU placement/cache locality as a confound, rather than
a universal notification penalty. Original failed rounds did not record thread
placement, so their exact assignments cannot be reconstructed. Frequency effects
were not independently isolated.

### Adoption decision and integration

On 2026-09-19, the user approved notification-based wakeup with the small GPU
cases treated as documented limitations, not adoption blockers. This changes the
acceptance policy; it does **not** turn the original failed nonregression gate
into a pass. A full matched-placement performance study is no longer a required
adoption gate. The targeted study still does not establish general unbound
nonregression. Do not hard-code application CPU affinity in CubeCL.

[CubeCL PR #19](https://github.com/tensor4all/cubecl/pull/19) merged as
`a2adda17affd40494393a1f40d90980e1235617c` after all six final CI jobs passed.
Cubek [PR #13](https://github.com/tensor4all/cubek/pull/13) also merged on its
existing 0.2 release line to align the CubeCL runtime types. The tenferro dependency
update and materialization repair merged in
[PR #1820](https://github.com/tensor4all/tenferro-rs/pull/1820) as `d7a8c60c`, after
all required checks passed, including actual RunPod device execution. See
[integration validation](evidence/tenferro-integration-validation.md).
The post-merge eight-suite refresh is complete; see the snapshot below and the
[declared protocol](evidence/integration-refresh-protocol.md).
The PR candidate additionally registers the wake target before returning a client
and tests delayed initialization plus concurrent publishers: 46 common unit tests
and all-target common Clippy pass locally. Hosted code-quality/documentation also
pass. The CUDA container lacks a Vulkan adapter, but both hosted Linux versions
now pass after switching to GitHub-hosted runners. An initially green Miri job
had skipped all tests due to the fork's package naming; direct package selection
corrects this, and both local and final hosted Miri actually pass 43 tests.
All six final hosted checks passed before the merge. The full CUDA library suite reproduces one existing f16
reinterpretation failure on both main and the candidate. See the
[PR validation checkpoint](evidence/cubecl-pr-validation.md) for exact outcomes.
Loom was not run.
The historical 44-test candidate and archived patch below are not this final PR
revision. The refresh below measures the integrated revision, not an isolated
notification-only change.

### Post-merge refresh snapshot

With tenferro `d7a8c60c`, the `20260919_142600` BMM medians are 2.152053 ms
(trace), 2.167014 ms (eager), and 2.079491 ms (PyTorch). The tenferro medians are
5.9%/6.5% below the historical published run, but this is not a randomized paired
causal comparison. Host-side overhead remains; no universal latency guarantee
or single-wait-per-call claim follows.

All eight suites have fresh results: 201 numerically checked successes,
61 unsupported and five not-configured rows. The two AD suites were corrected
and rerun at `20260919_151433` after discovering legacy execution-only checks;
see [small AD notes](linalg_ad_latency.md) for retained small-case regressions.
Provider threads are 1; PyTorch intra-op/inter-op are both 1. CPU affinity remains
unrestricted (CPUs 0–63), unlike the earlier matched-L3 diagnostic experiment.
[Refresh manifest](evidence/integration-refresh-summary.json) records exact
sources, raw-file hashes, historical ratios and limitations.

### Evidence and handoff

[Evidence index and reproduction conditions](evidence/README.md) includes the
candidate patch, declared protocols, timing/IQR summaries and profile summaries.
The patch is a diagnostic candidate against the pinned Tensor4all CubeCL fork,
not an accepted upstream fix. Large raw traces and executables were retained
locally under `data/diagnostics/bmm-sleep/`; they are not included in this notes
bundle. [Related elementwise analysis](elementwise.md).
