# Runtime compression candidate after the Ubuntu migration

The remaining objective is another 10% reduction in whole paid Pod lifetime
through startup and environment preparation, using the public official runner.
The successful Ubuntu 24.04 main run spends 26 of 320.178 seconds in runtime
setup, above the existing 5% need threshold. Its 17 seconds of runtime downloads
also show why a small byte improvement cannot by itself establish that objective.

Before measurement, commit `1ea15d4` fixed one offline comparison of zstd 3 and
10 on the complete common and both CUDA SDK tar streams from run 38051889375.
The byte screen required at least 15% combined reduction for both runtime tiers,
with exact preservation of the whole decoded tar stream. It did not authorize
a timing claim or paid allocations.

The screen failed: controlled reductions were 11.55% for CUDA 12.6 plus common
and 10.78% for CUDA 12.8 plus common. All decoded stream hashes matched. No
production change or paid comparison follows this candidate. Do not select a
different level or relax the threshold to turn this result into a pass. Local
compression/decompression times are operational diagnostics only.

[Full results and protocol](../../result/nvidia-gpu/ci/runtime-compression-screen.md)
retain every case, original and controlled archive size, checksum, reproduction
setting, and stage-cost evidence. The old bootstrap campaigns remain
INCONCLUSIVE; their driver-identity failures and retry ceiling are unchanged.
A separate candidate with enough startup/preparation headroom is still needed.
