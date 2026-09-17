# Native 1T exploratory comparison — INCONCLUSIVE

User authorized a distant idle L3 domain with workloads elsewhere. All four
variants used CPU32, explicit kernel/provider1T, the same OpenBLAS0.3.26 shared
library, three warmups and15 samples, sequentially. Thread checks and source
inspection confirm RAYON_NUM_THREADS feeds CpuContext; OpenBLAS and PyTorch
intra/inter-op getters report1. No thread-verification preload was used during
timing (the GEMV experiment alone uses its diagnostic dispatch preload).

Both completed suites are **INCONCLUSIVE** under the predeclared interference
gate: the first observed Julia and pi on the selected L3; the second observed
pi. Julia affinity was0–63, so initial distance did not prevent migration.
CPU24 and CPU40 preflights failed and no timing was collected there. A fresh
all-domain preflight selected CPU32 again for the single complete rerun. No
favorable case was selectively retried. The monitor also sometimes classified
short-lived benchmark workers as foreign; the independently identified Julia/pi
samples suffice to invalidate both suites regardless of those false positives.
The executable-name heuristic subsequently added to `run.py` was not used in
these recorded suites and does not retroactively change their classification.

## LM opt_flops: raw medians (milliseconds), not accepted speedup evidence

| Suite | Baseline 2fd8163 | Copy636cd499 | Copy + GEMV probe | PyTorch |
|---|---:|---:|---:|---:|
| Initial CPU32 |129.066|134.049|130.819|114.021|
| Full rerun CPU32 |140.396|127.012|123.093|105.735|

Copy's apparent change reverses direction between suites (about3.9% slower vs
9.5% faster). GEMV's apparent incremental reduction is only about2.4–3.1%; its
prior instruction result was4.3%, below the5% instruction gate. Neither trial
justifies production promotion or an advertised speedup. GEMV remains a
standalone diagnostic, with96 independent nonzero correctness checks passed.

All cases, both strategies, medians/IQRs, commands, exit statuses, timestamps,
CPU activity and process observations are in `summary.json`, the raw logs and
`activity.json` / `runs.json`; `summarize.py` regenerates the summary. Controls'
identical second strategy reuses the first measurement and is not an independent
sample. The existing harness retains aggregate median/IQR, not individual sample
times. These are warmed eager-workflow times, including intrinsic dispatch and
operand-handle work, not prepared standard-operation kernel-only latencies.
Input creation, runtime/path setup, warmup and result destruction are outside
its timer. Native inputs are zeros: separate unit/AD and nonzero probe validation
must not be mislabeled a full nonzero LM cross-backend oracle.

No other workload was stopped or re-pinned, and no shared service was changed.
The maintainer authorized temporarily excluding CPU32–39 from competing jobs,
but the first affinity write failed with EPERM (the Julia jobs belonged to a
different UID); passwordless sudo was unavailable. No measurement started.
Independent verification found all 369 live-thread masks unchanged; see
`reserved/post-failure-verification.json`. The attempted script's restoration
warning concerned redundant writes to unchanged masks, not actual affinity loss.
No privilege workaround or further timing retry was attempted.

The maintainer then stopped measurement attempts and requested PR integration.
Previously measured instruction savings remain instruction results, not native
speedup claims. Acceptance-quality timing and 4T remain deferred.
