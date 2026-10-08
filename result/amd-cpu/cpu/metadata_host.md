# CPU metadata: Host borrowed views (#2040)

Run `20261008_host_views`, AMD Ryzen 9 9955HX, Linux MKL devcontainer, CPU **1 thread**, no external affinity pinning. Pure metadata needs no session or BLAS work. tenferro-rs is explicitly pinned to `b3f47296244ff7b7c55ac0a75f782cb0835418c1`, as requested in the [maintainer reply](https://github.com/tensor4all/tenferro-rs/issues/2040#issuecomment-6057395295); this is not a latest-main run.

[MWE source](https://github.com/tensor4all/tenferro-benchmark/blob/1415ab1b1decc2bf93da6e31938909bd03d7e4e4/mwe/cpu_followup/src/metadata_host.rs), [predeclared experiment](../../../data/results/amd-cpu/cpu/metadata_host/20261008_host_views/declaration.json), [paired summary](../../../data/results/amd-cpu/cpu/metadata_host/20261008_host_views/summary.json), [environment/providers](../../../data/results/amd-cpu/cpu/metadata_host/20261008_host_views/run_t1.yaml), [lossless process records](../../../data/results/amd-cpu/cpu/metadata_host/20261008_host_views/processes.jsonl.gz). The original [TensorValue evidence](followup_12x.md) remains valid and unchanged.

## Timing and lifetime contracts

Every batch constructs separate, equivalent column-major f64 owners outside the timer. All typed `as_view()` inputs and output-retention capacity are prepared before timing. The timer includes only the declared metadata transformation and storing each returned descriptor. Input generation, wrapping, input-view construction, validation and destruction are untimed. All outputs are retained through clock stop. For typed arms, every owner is retained until after all its input/output views are destroyed; for TensorValue the consuming output retains the underlying owner.

The typed arms borrow `TypedTensorView<f64, R, Host>`; the original arm consumes owned, dtype-erased, dynamic-rank `TensorValue`. Their ownership/lifetime contracts differ and their costs must not be treated as interchangeable. Host DynRank keeps dynamic input/output rank. Static slice uses Rank<1>, and static transpose uses Rank<2>, preserving rank. Static-input reshape uses Rank<1> but returns **DynRank**, not Rank<2>. Metadata descriptors match Julia; every typed logical output element and its address within the original owner are checked untimed, confirming aliasing rather than a copy.

Each process uses 3 warmups and 15 batched samples. Four comparison rounds run Julia → owned → Host DynRank → Host static, then the reverse, reverse, forward. Each Rust arm has four independent A/A pairs in AB/BA/BA/AB order. Gates declared before collection: median paired ratio ≥1.2, every A/A pair within 10%, maximum per-process sample CoV ≤20%, correctness passed. Target interval is 50 ms; the 512 MiB retained-memory budget and 2,000,000-operation cap can make batches shorter. Values are batched throughput normalized per operation, **not isolated call latency**. Julia transpose specializes the fixed permutation and can reduce the loop to compact metadata stores; its sub-nanosecond normalized time is not a physical single-call latency.

## Paired comparisons with Julia

Times are medians of four process medians, in ns/op. Ratios are medians of the four paired round ratios; they need not equal the quotient of the displayed aggregate times.

| operation / fixture | Rust representation | Rust ns/op | Julia ns/op | Rust / Julia | paired ratio range | max A/A deviation | max CoV | conclusion |
|---|---|---:|---:|---:|---:|---:|---:|---|
| reshape [1024] → [32,32] | TensorValue (consuming owned) | 336.726 | 7.766625 | 43.267× | 39.874–43.457× | 5.8% | 16.9% | confirmed-slower |
| reshape [1024] → [32,32] | Host borrowed / DynRank | 365.380 | 7.766625 | 47.085× | 42.362–48.227× | 4.5% | 16.9% | confirmed-slower |
| reshape [1024] → [32,32] | Host borrowed / static input rank | 316.467 | 7.766625 | 40.761× | 36.857–41.034× | 2.6% | 16.9% | confirmed-slower |
| slice [4096], 128:3968:2 → [1920] | TensorValue (consuming owned) | 278.965 | 2.335538 | 120.291× | 112.209–125.362× | 6.6% | 15.1% | confirmed-slower |
| slice [4096], 128:3968:2 → [1920] | Host borrowed / DynRank | 292.649 | 2.335538 | 124.890× | 123.265–132.593× | 1.8% | 15.1% | confirmed-slower |
| slice [4096], 128:3968:2 → [1920] | Host borrowed / static input rank | 226.906 | 2.335538 | 96.700× | 95.555–100.886× | 3.0% | 15.1% | confirmed-slower |
| transpose [2,2], axes [1,0] | TensorValue (consuming owned) | 214.643 | 0.735334 | 293.049× | 255.186–327.567× | 5.3% | 14.8% | confirmed-slower |
| transpose [2,2], axes [1,0] | Host borrowed / DynRank | 324.871 | 0.735334 | 444.300× | 393.051–492.424× | 3.2% | 14.8% | confirmed-slower |
| transpose [2,2], axes [1,0] | Host borrowed / static input rank | 242.615 | 0.735334 | 331.828× | 291.647–372.086× | 9.7% | 14.8% | confirmed-slower |

## Representation comparison within each balanced round

Owned / Host >1 means the borrowed Host arm took less time. These compare the explicitly different contracts above. The static slice improvement passes the same 1.2 threshold and both Rust arms’ noise gates; small differences below that threshold are not confirmed improvement claims.

| operation | owned / Host DynRank | owned / Host static input rank |
|---|---:|---:|
| reshape [1024] → [32,32] | 0.920× | 1.064× |
| slice [4096], 128:3968:2 → [1920] | 0.943× | 1.239× |
| transpose [2,2], axes [1,0] | 0.657× | 0.878× |

Host specialization does not close the Julia gap in these fixtures. Static-rank slicing improves relative to TensorValue, while reshape and transpose do not show a ≥1.2 improvement over TensorValue. These measurements narrow the discussion to the measured view paths; they do not identify shared internal fields as the cause or establish that a broader representation redesign is needed.

## Batch sizes and measured interval

Ranges below cover the four comparison processes per arm; durations are their median measured batch durations.

| case | arm | operations per sample | batch duration (ms) |
|---|---|---:|---:|
| `metadata.reshape_view` | julia-base | 63,550–63,550 | 0.492–0.543 |
| `metadata.reshape_view` | metadata | 57,456–57,456 | 19.243–19.581 |
| `metadata.reshape_view` | metadata-host-dyn | 54,032–54,032 | 19.508–20.257 |
| `metadata.reshape_view` | metadata-host-static | 55,097–55,097 | 17.356–17.508 |
| `metadata.slice_view` | julia-base | 16,256–16,256 | 0.037–0.039 |
| `metadata.slice_view` | metadata | 15,827–15,827 | 4.225–4.554 |
| `metadata.slice_view` | metadata-host-dyn | 15,556–15,556 | 4.531–4.668 |
| `metadata.slice_view` | metadata-host-static | 15,701–15,701 | 3.486–3.636 |
| `metadata.transpose_view` | julia-base | 1,864,135–1,864,135 | 1.255–1.529 |
| `metadata.transpose_view` | metadata | 262,144–262,144 | 54.875–57.798 |
| `metadata.transpose_view` | metadata-host-dyn | 262,144–262,144 | 84.459–86.886 |
| `metadata.transpose_view` | metadata-host-static | 262,144–262,144 | 62.715–65.653 |

## Reproduction

These are the recorded collection commands. To collect fresh results, check out the linked MWE source commit and the pinned library commit, rebuild, and use a new output directory. Use the default Linux MKL devcontainer. The MWE’s private `system-mkl` feature maps to the pinned upstream `blas` API and links the installed MKL; it is separate from the root harness’s migrated CPU feature names.

```bash
docker exec -u vscode -w /workspaces/tenferro-benchmark 14099c68ff8e bash -lc '
  CARGO_TARGET_DIR="$PWD/target/followup-b3f47296" CARGO_BUILD_JOBS=8 \
    cargo build --release --locked --manifest-path mwe/cpu_followup/Cargo.toml'
python3 mwe/cpu_followup/collect_metadata_host.py --output data/results/amd-cpu/cpu/metadata_host/20261008_host_views
python3 scripts/format_metadata_host_results.py --raw data/results/amd-cpu/cpu/metadata_host/20261008_host_views
```

The collector sets OMP/MKL/Rayon/OpenBLAS/Julia thread counts to 1 via `configure_cpu_thread_env 1`, sets `CPU_FOLLOWUP_TARGET_NS=50000000`, and exports the installed MKL/compiler library paths. Host and container idle guards remain enabled. Exact process commands, timings and signatures are in the compressed records; [restore instructions](../../../mwe/cpu_followup/README.md) apply unchanged. Both typed arms and the existing Julia reference are registered under #2040 in `cpu/perf_issues` (manifest v5).
