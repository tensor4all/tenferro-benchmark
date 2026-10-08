# Compact runtime preparation diagnostic

CUDA setup occupied 294 seconds of the 920-second production diagnostic;
its 1.15 GB compressed cache timed out after three minutes before falling
back to installation. Prepare complete minimal CUDA runtime libraries and JIT
headers on a hosted runner, using reviewed root helper revision f3b995657adbbcabc636c5377877205d920a35aa.
Bundle that tree with cuTENSOR, PJRT wheels and real Cargo/nextest binaries
before GPU allocation. Test source and library binaries remain frozen at
2605c46f. Native CUDA 12.8 image is diagnostic only; production must retain
its CUDA 12.6-compatible path. Hosted preparation compiles the CubeCL-style
headers with real NVRTC before uploading. No GPU allocation in the initial
prepare-only run. Complete GPU correctness and paid-time effects are unverified;
this does not promote the compact seed or reuse failed confirmation samples.

After broad live pricing reported A40/A5000/2000 Ada out of stock and
6000 Ada stock at $0.99/hour, permit that reviewed model for diagnostics
with a $1.10/hour ceiling. This is not a replacement confirmation sample;
new runtime GPU correctness must be established before a separately declared
complete paired campaign. Paid time remains primary; higher cost is visible.
