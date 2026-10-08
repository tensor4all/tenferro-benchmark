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

Dual-SDK hosted run 37814278025 succeeded at frozen controller
dba171d742d8aff450ea9479706870ecd9456f46 with no GPU allocation.
Both CUDA 12.6 and 12.8 seeded trees passed real NVRTC header compilation.
Five-part SDK uploads total 656,567,385 bytes for 12.6 and 1,074,447,802
bytes for 12.8; common execution/test payload totals 1,782,795,429 bytes.
Only the driver-selected SDK is transferred to the GPU. Diagnostic run
37814911641 reuses these exact artifacts and requests A4000 at the existing
$0.35/hour ceiling. Baseline hosted preparation is run 37814618112.
These preparations are not paid-time samples or evidence of a speedup.

Baseline hosted preparation 37814618112 succeeded. A4000 diagnostic
37814911641 stopped at the public stock check before creating a pod; fee
was zero. Public pricing then showed only A40 in stock among the seven
reviewed diagnostic models, at $0.59/hour (Low stock), within its existing
$0.60/hour ceiling. A fresh A40 diagnostic reuses the candidate artifacts;
this is not a selective replacement in a confirmation campaign.

A40 diagnostic 37815639364 succeeded: all 285 CUDA and 3 PJRT tests plus
tutorial passed with actual NVRTC 12.8, and pod deletion was confirmed.
Accepted pod paid time was 452.517209 seconds at $0.59/hour (~$0.07416).
This validates the full compact SDK on a GPU, but is not confirmation.
A new campaign freezes controller dba171d742d8aff450ea9479706870ecd9456f46,
A40 ($0.60/hour ceiling), NVRTC 12.8, both hosted preparations, and three
pairs in B/C,C/B,B/C order. First baseline is run 37816754595. All six
must pass the complete workload; every pair must be nonregressing, median
paid time must reduce at least 20%, and each arm's max/min must be <=1.5.
Any failure makes the entire campaign inconclusive, without replacements.
Runtime preparation is identical in both arms: the isolated effect is the
SVD oracle change, not a claim about runtime staging versus production.

Campaign 3's first complete pair passed every test and deletion: baseline
37816754595 paid 547.591566 seconds; candidate 37818003215 paid
390.447417 seconds (descriptive 28.6973% reduction). Both were A40 with
NVRTC 12.8 at $0.59/hour. This one pair is not promotion evidence;
second-pair candidate 37818887742 is running and full confirmation remains
pending. A transient API 404 immediately after dispatch was re-polled on
the same run handle, which was confirmed live; no replacement was created.

Campaign 3 stopped INCONCLUSIVE after baseline 37819774823 failed with
HTTP 500 / no instances available before any pod creation. Candidate
37818887742 had passed the complete workload and deletion in 390.088442
seconds, but is retained within the failed campaign, not reused.

The next controller permits at most three create attempts for the same
GPU, retrying only the exact no-instances HTTP 500 when no pod was created.
Each attempt receives a unique runner label. A created-pod failure, other
error, or exhausted limit stops the run and invalidates the entire campaign.
This handles public-price/stock races before the paid window, not paid
sample replacement. All attempt logs remain visible. Fresh pods, frozen
sources and SDK, complete workload, pair order and acceptance gates stay
unchanged. The next campaign starts anew with no earlier samples.
