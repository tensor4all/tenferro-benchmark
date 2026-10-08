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

Hosted-only run 37808424802 succeeded, including real NVRTC header
compilation and five uploaded parts totaling 2,857,196,709 bytes. The earlier
headers-only prepared payload was 1,786,195,109 bytes. Complete CUDA libraries
add about 1.07 GB of transfer, so staging is not assumed faster without timing.
6000 Ada diagnostic 37808857457 reported out of stock before creation, with
no GPU charge. Runtime correctness on a GPU is still pending.

A4000 diagnostic 37812347316 also failed before pod creation with no charge.
The prototype restricted drivers to CUDA 12.8+, unlike production's 12.6
floor; this could narrow eligible capacity, but does not establish the cause
of stock failures. The next diagnostic uses digest-pinned NVIDIA CUDA
12.6.3 runtime on Ubuntu 24.04 (Python 3.12 and hosted tool compatibility),
retaining pre-registration compile/load/launch validation. Hosted preparation
creates separately split minimal SDKs for 12.6 and 12.8 and checks each with
real NVRTC. GPU setup selects the driver-compatible SDK and transfers only
that SDK. New artifacts are mandatory: the previous prepared run lacks
these separate SDK artifacts. All GPU tests and 12.8 capability selection
remain intact. Correctness and paid-time improvement remain unverified.
