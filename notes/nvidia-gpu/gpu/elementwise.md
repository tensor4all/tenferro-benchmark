# NVIDIA GPU elementwise: item notes

Analysis date: 2026-09-19. A100 80GB PCIe; benchmark `9a5b104`, tenferro
`40e24634` plus the explicitly identified experimental materialization repair.
These are diagnostic comparisons, not replacements for published timing rows.

<a id="elementwise_chain_f64_1m"></a>
## f64 chain, 1,048,576 elements

The expression repeats `tanh(t * a + b)` eight times.

- Original eager: 24 arithmetic kernels plus **40 redundant materialization
  kernels per expression**, introduced by compact borrowed inputs being packed
  before owned-only CUDA entry points.
- The repair launches existing kernels directly for supported zero-offset,
  column-major compact reads. Separate ten-expression Nsight profiles retain
  80 add, 80 multiply and 80 tanh kernels on both versions; `tensor_permute`
  drops from 403 to 3. Those three remaining copies are one-time/setup work.
- Three unprofiled paired rounds (explicit provider 1T, sequential, 5 warmups,
  30 samples) improve eager from 1.05–1.66 ms to 0.45–0.59 ms, or 2.35–3.09x.
  All 36 numerical records and the declared trace/BMM nonregression checks pass.
- Trace fuses the arithmetic into one kernel and is about 0.15–0.17 ms in the
  repair comparison. The previous Torch **eager** measurement was about 0.32 ms;
  Torch was not rerun in those pairs and `torch.compile` was not compared.

The patch does **not** remove every materialization. Offset/noncompact views,
scalar/broadcast binary inputs, unlisted operations and most view-output paths
retain fallbacks. CPU common-linalg routing also uses existing read hooks; no
CPU speedup was measured. The repair was uncommitted/unmerged at this analysis.

<a id="elementwise_chain_f64_1k"></a>
## f64 chain, 1,024 elements

This is a per-call latency diagnostic, not shared-session throughput. The same
materialization repair improves eager with paired ratios 0.206–0.323. Remaining
latency includes host dispatch, output allocation, service coordination and
24 kernel launches; trace also benefits from eliminating intermediate device
memory traffic through fusion.

Small eager measurements are sensitive to caller/device-service CPU placement.
A subsequent CubeCL wakeup candidate initially failed one chain comparison by
20.2%, but four same-L3 diagnostic pairs have ratios 0.971–1.023. See the
[dense BMM note](dense.md#dense_batched_matmul_f64_b1024_256) for the distinction
between that diagnostic and the still-failed original acceptance gate. Neither
GPU kernel slowdown nor a universal notification penalty was established.

## Evidence

[Evidence index](evidence/README.md): source patch and paired medians/IQRs.
Original local profiles and raw records are under
`data/diagnostics/read-path-results/`; large traces/binaries are not included in
this notes bundle. Follow-up CPU-affinity findings concern the separate CubeCL
wakeup candidate, not the already measured copy-removal effect.
