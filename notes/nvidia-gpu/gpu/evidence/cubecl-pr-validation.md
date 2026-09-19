# CubeCL notification PR validation checkpoint

2026-09-19, PR https://github.com/tensor4all/cubecl/pull/19,
head `966a6d307f84a6bca4ac4f2588f41c90a0f5c62d`, based on main
`1c9be425daf5a66c9829f4f9ae0bc22471624567`.

The final candidate registers the wake thread before exposing a client as well
as before the server's first queue check. This strengthens initialization
ordering beyond the historical diagnostic patch. Five focused tests cover
blocked initialization, concurrent producers, idle full/partial buffers,
pre-registration publication and notification before park.

## Passed

- Common library: 46 tests, one test thread; common all-target Clippy with
  `-D warnings`; workspace formatting.
- Negative control: removing client-side registration makes its blocked-init
  test fail. Restoring the candidate makes that test pass again.
- Repository xtask audit, workspace lint, typos, CI build, documentation and
  doctests. Typos needed a narrowly scoped allowance for the pre-existing CUDA
  descriptor identifier `cpy` with the current typos tool. The initial LLVM
  setup failed because the container's cache parent is root-owned; a writable
  `tracel` cache subdirectory resolved that setup failure.
- CUDA synchronization smoke tests: 2/2.
- Hosted prepare-checks, code-quality and documentation on the exact PR head.

## Failed / unavailable — not represented as successful

- Repository xtask CI tests cannot complete in the CUDA container: no Vulkan
  adapter. WGPU reports 5 passed, 371 failed, 15 ignored after adapter creation
  fails. No Vulkan or Miri success is claimed.
- Full CUDA library tests: 711 passed, 1 failed, 24 ignored. Failure:
  `tests::test_cubecl_std::reinterpret_slice_f16::global::read_from_i8x4` gives
  `[1.0, 0.0]` instead of `[1.0, -8.5]`. Three baseline and three candidate runs
  reproduce the same failure in every run. Baseline restores the entire channel
  file to main 1c9be425; the other sources and dependency lock stay unchanged.
  Candidate source was restored and byte-compared with the PR worktree afterward.
  This isolates an existing failure, not a new wakeup regression; no unrelated
  reinterpretation change was made.
- Hosted Linux stable/previous and nightly Miri jobs are queued with upstream
  GCP runner labels. The repository API reports zero runners. They have not run,
  have not been cancelled and have not been bypassed.

CI run: https://github.com/tensor4all/cubecl/actions/runs/35441669193

## Runner correction and non-vacuous Miri verification

The user explicitly requested fixing CI configuration rather than skipping
checks. Head `62ff7e2aa1eaadac65aa25fc6f64e1bd957da594` changes the Linux/Miri
jobs to GitHub-hosted `ubuntu-24.04`. Both Linux versions completed successfully:
19 test summaries / 1,521 passing test executions each (including repeated target
configurations); all five notification tests actually executed. The earlier
runner-blocked run was superseded automatically by workflow concurrency.

The Miri job initially reported success but ran **zero tests**. `tracel-xtask`
4.16 derives package names from directory basenames, while this fork renamed
Cargo packages to `t4a-*`. The Miri `only` filter therefore excluded every crate.
That green job is **not** evidence of Miri validation.

Head `7697cf4e070217369a2a10ec1deca30fd2d37dea` selects `t4a-cubecl-common`
directly with Cargo Miri, preserving the original UB-only `-Zmiri-ignore-leaks`
mode and unit/integration target coverage. Local nightly 1.100.0 (2026-09-19)
Miri actually executed **43 tests, all passing**, including all five notification
tests. Three native tests are excluded by existing Miri cfgs. The container
needed a writable XDG cache directory for Miri's sysroot; the initial failed
setup log is retained. Final hosted run:
https://github.com/tensor4all/cubecl/actions/runs/35443446621

## Final CI and merge

Run 35443446621 completed with **all six jobs successful** on head
`7697cf4e070217369a2a10ec1deca30fd2d37dea`. Logs confirm 43 actual Miri tests
passing in each unit/integration invocation, including all five notification
tests. Linux stable and previous, code-quality and documentation also pass.

PR #19 merged on 2026-09-19T12:48:59Z as
`a2adda17affd40494393a1f40d90980e1235617c`. Merge state and commit were checked
through the GitHub API. The earlier queued/skipped runs above remain historical
evidence, not the final validation state.

## Remaining / reproduction evidence

Update and merge tenferro-rs with this merged dependency, then follow the
integration refresh protocol. Small GPU
latency regressions were accepted separately and do not waive correctness or
notification safety.

Local source mirror: `data/diagnostics/cubecl-pr-src/`.
Logs: `data/diagnostics/cubecl-pr-*.log`, including full failed gate logs and all
six reinterpretation reproductions. These large build logs are local artifacts;
this checkpoint and the PR body preserve their results durably. The archived
historical performance patch/hashes remain unchanged and must not be mistaken
for the final PR revision.
