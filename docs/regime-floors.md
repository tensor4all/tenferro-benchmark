# Regime Floors: What May Be Called `throughput`

A case's **regime** is its workload class: `throughput` (compute- or
bandwidth-bound) or `latency` (dominated by per-call launch, dispatch and
synchronization). The dashboard publishes one page per regime and never mixes
them, so the regime decides which page a case's bars appear on.

Two rules keep that page honest. Both are enforced at load time, in
`scripts/regime_policy.py`, by `scripts/generate_dashboard.py` and by
`scripts/validate_benchmark_suite.py`; both fail closed.

## 1. No implicit regime

A case's regime comes from its own `regime` or from its suite's
`defaults.regime`. There is **no fallback**. Before this rule the dashboard
defaulted an undeclared case to `throughput`, so every undeclared suite (13 GPU
suites, including `gpu/linalg_jvp_vjp` at n=256) published its cases as
throughput — including cases that measure per-call framework overhead.

Where a case lives:

| declaration | where |
|---|---|
| inline problems (`benchmarks/gpu/*.yaml`) | `regime:` on the problem |
| catalogue suites (`problems: {source, include}`) | `regime:` on the instance record in `data/instances/*.json` |
| suite-wide | `defaults.regime` in the suite YAML |

Declaring a case `latency` is not a demotion of its numbers: it moves the row to
the page that describes what it measures.

## 2. A throughput case must be large enough to be one

The case's **declared working set**, in scalar elements, must reach its family
floor. The working set is derived from the declaration (`n` for a 1-D
elementwise case, `prod(shape)` for an instance record, `m*n` for a matmul,
`batch*m*n` for a batched matmul, `nnz` for a sparse case, an explicit
`size_elements` where the shape lives outside the declaration). A case that
declares `throughput` below its floor is a load error:

```
gpu/dense_throughput: case 'elementwise_chain_f64_1k_throughput' declares
`throughput` at 1,024 elements, below the elementwise floor of 262,144 elements.
At this size the measurement is per-call framework overhead: declare the case
`latency` (see docs/regime-floors.md).
```

### How a floor is derived

A throughput case must be on the linear part of its cost curve: doubling the
work has to move the measured time. Fitting the published medians of one family
to `t = F + c·n` (F = per-call fixed cost, c = marginal cost per element), the
time gained by doubling `n` is `c·n`, and the case stops being a constant when
that is at least 10% of its own time:

```
c·n ≥ 0.1·(F + c·n)   ⇔   n ≥ F / (9c)
```

The evidence column below names the size pair that fixes `F` and `c` for each
family. Where only one size was ever published, the floor sits at that size and
is marked **provisional** — conservative, not measured.

### The table

| family | floor (elements) | provisional | evidence |
|---|---|---|---|
| `dense` | 65,536 | no | `dense_svd_f64_256` is 65,536 elements and takes 12.0 ms, ~900× the 13 µs dispatch scale, so a 256² factorisation is compute; `dense_batched_matmul_f64_b16_32` is 16,384 elements and takes 13 µs, so it is overhead. |
| `einsum` | 1,048,576 | yes | the only published einsum case is a 3072³ binary contraction (9.4e6 elements, 3.17 ms). |
| `elementwise` | 262,144 | no | the two `elementwise_chain` points fit `t = 69.6 µs + 27.9 ps·n` (1k = 69.6 µs, 1M = 99.6 µs in deferred-batch mode): 1000× the elements buys 1.43× the time, and `n ≥ F/(9c) ≈ 2.8e5`. At 1M the fixed share is still ~70%: the floor is a minimum, not a promise of purity. |
| `fft` | 1,048,576 | yes | only the 1M point is published (0.063–0.135 ms); the 1k point was never measured. |
| `linalg` | 65,536 | yes | cholesky 32² (1,024 elements, 61 µs) and 512² (262,144 elements, 371 µs) put the crossing near 5.7e3 elements; 2^16 keeps every 512² case and excludes the 32²/64×32 catalogue entries. The public API's fixed cost is ~60 µs, so a 512² call is still mostly overhead — hence *provisional*. |
| `linalg_ad` | 1,048,576 | no | QR JVP at n=256 (65,536 elements) takes 1.43 ms while the 256² factorisation it wraps costs tens of µs, and n=4 already costs 0.37 ms: 64× the size buys 4× the time. No AD case is measured above 512², so the floor is the conservative 2^20 and **no AD case currently clears it** — that is the finding, not a gap in the table. |
| `permutation` | 1,048,576 | no | every published pattern is 3.9e8–5.4e8 elements (5.0–8.4 ms). |
| `sparse` | 1,048,576 (nnz) | yes | spmv at 6.7e7 nnz takes 2.26 ms and spmm at 4.2e6 nnz takes 16.3 ms; 2^20 is two orders below the smallest published case. |
| `tensornetwork` | 1,048,576 | yes | the single case is a whole-network contraction at 78 ms. The fixture is an external submodule, so its element count is not derivable; the case declares `floor_exempt` with the reason instead of inventing a size. |

`perf_issues` is a diagnostic catalogue with `defaults.regime: latency`; it has
no throughput case and therefore no floor.

### What the floors moved

Nothing was deleted. Every case below keeps its id, its numbers and its bars;
only its page changed:

| suite | case | elements | was | now |
|---|---|---|---|---|
| `gpu/linalg_jvp_vjp` | 10 × `linalg_ad_large_grad_sum_*_{jvp,vjp}_f64_{256,512}` | 65,536 / 262,144 | throughput (undeclared) | latency |
| `gpu/linalg_ad_throughput` | `linalg_ad_large_grad_sum_qr_jvp_f64_256`, `..._qr_vjp_f64_256` | 65,536 | throughput | latency |
| `gpu/dense_throughput` | `dense_batched_matmul_f64_b16_32_throughput` | 16,384 | throughput | latency |
| `gpu/dense_throughput` | `elementwise_chain_f64_1k_throughput` | 1,024 | throughput | latency |
| `gpu/reuse_contraction` | 12 × `dot_general_*_s256`, `einsum_plan_*_s256` | ≤ 65,536 | throughput | latency |
| `gpu/structural` | `embed_diag_default_f64_n1024` | 1,024 | throughput | latency |
| `gpu/*` catalogue | 439 records at 1,024–131,072 elements | below floor | inherited suite default | `latency` on the record |

`gpu/linalg_ad_throughput` keeps its suite id and its deferred-batch timing mode
(`defaults.measure: deferred_batch`) for trend continuity: the suite id names the
collection, `regime` names the workload class, and the AD runner selects its
route from `measure` (falling back to `regime` for suites that do not declare
one). Its page says in words that it has no throughput-regime case.

## Timing mode is not regime

| field | meaning | who reads it |
|---|---|---|
| `regime` | workload class → page | dashboard, validator |
| `measure` | timing route: `deferred_batch` (one retirement per K calls) or `per_call` | the AD runner |

A case can be `latency` and `deferred_batch`: that is exactly what the two
`gpu/linalg_ad_throughput` cases are, and the suite page labels the amortization
in the latency section.

## Adding a case

1. Declare `regime` (on the problem or the instance record) or rely on an
   explicit `defaults.regime` in the suite. Never leave it out.
2. If it is `throughput`, check its working set against the table. Below the
   floor, declare it `latency` — or, if you believe the floor is wrong, change
   the floor *with* evidence in the same pull request.
3. `uv run python scripts/validate_benchmark_suite.py benchmarks/gpu/<suite>.yaml`
   fails on both mistakes.
4. If the suite is generated (`benchmarks/gpu/linalg_jvp_vjp.yaml`,
   `benchmarks/gpu/linalg_ad_latency.yaml`), change its generator, not the YAML:
   `run_gpu_linalg_jvp_vjp.sh` rewrites both from their seeds before every
   collection, so a hand edit is lost and a generator that declares nothing stops
   the run at load time. Those generators declare each case through
   `regime_policy.regime_for_case()`, which derives the regime from the case's
   size and its family floor; a size change then moves the case across the floor
   without a second edit.

## Out of scope: the CPU catalogues

The rule is enforced for `gpu/` suites. The CPU catalogues still mix size
classes (their `throughput` default is marked provisional in each YAML) and
their floors are a follow-up: enforcing today's table on them would move several
hundred cases, which is a separate decision about what the CPU pages are for.
The CPU throughput banner therefore states the size range it actually holds and
omits the "at or above its family floor" claim.
