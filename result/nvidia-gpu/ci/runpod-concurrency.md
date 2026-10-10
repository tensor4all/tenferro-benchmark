# CUDA test concurrency comparison: INCONCLUSIVE

The single predeclared attempt stopped after its two serial A/A runs exceeded
its 1.15 max/min paid-lifetime limit. No two-process candidate ran. This does
not establish a speedup, a slowdown, or the safety of concurrent CUDA tests.
No slow run was discarded or replaced; no further Pods were allocated.

| Serial run | Paid lifetime | Runtime preparation | Archive transfer | CUDA tests | Peak sampled VRAM | Cost record |
|---|---:|---:|---:|---:|---:|---|
| [38066318486](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38066318486) | 601.490 s | 259 s | 30 s | 240 s | 1397 MiB | pre-delete |
| [38067070576](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38067070576) | 346.410 s | 27 s | 3 s | 240 s | 1397 MiB | startup fallback |

The A/A ratio was 1.7364 (73.64% spread relative to the faster run). Most of
the difference was preparation/transfer time; CUDA duration was identical.
Both runs used A40 in CA-MTL-1, CUDA 12.8, driver 570.195.03, and Intel Xeon
Gold 6342 CPUs on different provider machines. CPU affinity was not pinned.
Each passed the exact same 304 CUDA and three PJRT tests plus the tutorial,
and each Pod deletion returned HTTP 204. Total estimated GPU cost was
$0.1553502778, below the $1.50 cap. No paid Pod remains from this attempt.

The second pre-delete API read failed with HTTP 000 (DNS timeout), but the
new startup snapshot recovered price, start time and placement after successful
deletion. The telemetry fix therefore handled the real failure that had lost
the previous campaign's last observation. The first run used final metadata.
The memory monitor captured 239 samples per run at one-second intervals;
subsecond peaks and two-process memory requirements remain unmeasured.

Execution used harness `53b340d61ec05824b0118d06a342c05318aa2c0b`, trusted helpers
`15989736c9a8d0ac73c2348b833e4e143c9d4c2b`, and compiled source
`eb4d13fdfbd66541a6dfdc28f04d1cee0604f365` from artifact run 38056837195.
Both arms were defined to use the same pinned public official runner and Ubuntu
24.04. The [protocol](runpod-concurrency-protocol.json) was committed before
allocation. [Raw evidence](runpod-concurrency.json.gz) retains the bound protocol,
terminal state, full logs, jobs and cost records for both actual runs.

Retain serial production execution and land only the independently verified
telemetry repair. The earlier bootstrap campaign remains separately closed and
INCONCLUSIVE; none of its samples are reused here.
