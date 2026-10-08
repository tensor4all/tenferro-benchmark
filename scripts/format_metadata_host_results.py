"""Render #2040's pinned Host-view extension without overwriting original evidence."""
import argparse
import gzip
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'metadata': 'TensorValue (consuming owned)',
          'metadata-host-dyn': 'Host borrowed / DynRank',
          'metadata-host-static': 'Host borrowed / static input rank'}

def render(raw, output):
    declaration = json.loads((raw/'declaration.json').read_text())
    rows = json.loads((raw/'summary.json').read_text())
    processes = {}
    with gzip.open(raw/'processes.jsonl.gz', 'rt') as f:
        for line in f:
            record = json.loads(line)
            processes[record['file']] = json.loads(record['content'])
    prefix = '../../../' + raw.relative_to(ROOT).as_posix()
    code = declaration['harness_commit']
    lines = ['# CPU metadata: Host borrowed views (#2040)', '',
        f"Run `{raw.name}`, AMD Ryzen 9 9955HX, Linux MKL devcontainer, CPU **1 thread**, no external affinity pinning. Pure metadata needs no session or BLAS work. tenferro-rs is explicitly pinned to `{declaration['library_commit']}`, as requested in the [maintainer reply](https://github.com/tensor4all/tenferro-rs/issues/2040#issuecomment-6057395295); this is not a latest-main run.", '',
        f"[MWE source](https://github.com/tensor4all/tenferro-benchmark/blob/{code}/mwe/cpu_followup/src/metadata_host.rs), [predeclared experiment]({prefix}/declaration.json), [paired summary]({prefix}/summary.json), [environment/providers]({prefix}/run_t1.yaml), [lossless process records]({prefix}/processes.jsonl.gz). The original [TensorValue evidence](followup_12x.md) remains valid and unchanged.", '',
        '## Timing and lifetime contracts', '',
        'Every batch constructs separate, equivalent column-major f64 owners outside the timer. All typed `as_view()` inputs and output-retention capacity are prepared before timing. The timer includes only the declared metadata transformation and storing each returned descriptor. Input generation, wrapping, input-view construction, validation and destruction are untimed. All outputs are retained through clock stop. For typed arms, every owner is retained until after all its input/output views are destroyed; for TensorValue the consuming output retains the underlying owner.', '',
        'The typed arms borrow `TypedTensorView<f64, R, Host>`; the original arm consumes owned, dtype-erased, dynamic-rank `TensorValue`. Their ownership/lifetime contracts differ and their costs must not be treated as interchangeable. Host DynRank keeps dynamic input/output rank. Static slice uses Rank<1>, and static transpose uses Rank<2>, preserving rank. Static-input reshape uses Rank<1> but returns **DynRank**, not Rank<2>. Metadata descriptors match Julia; every typed logical output element and its address within the original owner are checked untimed, confirming aliasing rather than a copy.', '',
        'Each process uses 3 warmups and 15 batched samples. Four comparison rounds run Julia → owned → Host DynRank → Host static, then the reverse, reverse, forward. Each Rust arm has four independent A/A pairs in AB/BA/BA/AB order. Gates declared before collection: median paired ratio ≥1.2, every A/A pair within 10%, maximum per-process sample CoV ≤20%, correctness passed. Target interval is 50 ms; the 512 MiB retained-memory budget and 2,000,000-operation cap can make batches shorter. Values are batched throughput normalized per operation, **not isolated call latency**. Julia transpose specializes the fixed permutation and can reduce the loop to compact metadata stores; its sub-nanosecond normalized time is not a physical single-call latency.', '',
        '## Paired comparisons with Julia', '',
        'Times are medians of four process medians, in ns/op. Ratios are medians of the four paired round ratios; they need not equal the quotient of the displayed aggregate times.', '',
        '| operation / fixture | Rust representation | Rust ns/op | Julia ns/op | Rust / Julia | paired ratio range | max A/A deviation | max CoV | conclusion |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    fixtures = {'metadata.reshape_view':'reshape [1024] → [32,32]',
                'metadata.slice_view':'slice [4096], 128:3968:2 → [1920]',
                'metadata.transpose_view':'transpose [2,2], axes [1,0]'}
    for r in rows:
        ratios=r['round_ratios']
        lines.append(f"| {fixtures[r['case_id']]} | {LABELS[r['path']]} | {r['median_ns']:.3f} | {r['julia_ns']:.6f} | {r['ratio']:.3f}× | {min(ratios):.3f}–{max(ratios):.3f}× | {r['max_aa_deviation']:.1%} | {r['max_cov']:.1%} | {r['status']} |")
    lines += ['', '## Representation comparison within each balanced round', '',
        'Owned / Host >1 means the borrowed Host arm took less time. These compare the explicitly different contracts above. The static slice improvement passes the same 1.2 threshold and both Rust arms’ noise gates; small differences below that threshold are not confirmed improvement claims.', '',
        '| operation | owned / Host DynRank | owned / Host static input rank |',
        '|---|---:|---:|']
    for case in declaration['cases']:
        a={r['path']:r for r in rows if r['case_id']==case}
        lines.append(f"| {fixtures[case]} | {statistics.median(a['metadata-host-dyn']['owned_over_arm_round_ratios']):.3f}× | {statistics.median(a['metadata-host-static']['owned_over_arm_round_ratios']):.3f}× |")
    lines += ['', 'Host specialization does not close the Julia gap in these fixtures. Static-rank slicing improves relative to TensorValue, while reshape and transpose do not show a ≥1.2 improvement over TensorValue. These measurements narrow the discussion to the measured view paths; they do not identify shared internal fields as the cause or establish that a broader representation redesign is needed.', '',
        '## Batch sizes and measured interval', '',
        'Ranges below cover the four comparison processes per arm; durations are their median measured batch durations.', '',
        '| case | arm | operations per sample | batch duration (ms) |',
        '|---|---|---:|---:|']
    for case in declaration['cases']:
        for arm in declaration['arms']:
            ps=[processes[f'{case}-pair{i}-{arm}.json'] for i in range(4)]
            counts=[p['samples'][0]['iterations'] for p in ps]
            durations=[statistics.median(s['elapsed_ns'] for s in p['samples'])/1e6 for p in ps]
            lines.append(f"| `{case}` | {arm} | {min(counts):,}–{max(counts):,} | {min(durations):.3f}–{max(durations):.3f} |")
    lines += ['', '## Reproduction', '',
        'These are the recorded collection commands. To collect fresh results, check out the linked MWE source commit and the pinned library commit, rebuild, and use a new output directory. Use the default Linux MKL devcontainer. The MWE’s private `system-mkl` feature maps to the pinned upstream `blas` API and links the installed MKL; it is separate from the root harness’s migrated CPU feature names.', '',
        '```bash',
        'docker exec -u vscode -w /workspaces/tenferro-benchmark 14099c68ff8e bash -lc \'',
        '  CARGO_TARGET_DIR="$PWD/target/followup-b3f47296" CARGO_BUILD_JOBS=8 \\',
        '    cargo build --release --locked --manifest-path mwe/cpu_followup/Cargo.toml\'',
        f'python3 mwe/cpu_followup/collect_metadata_host.py --output {raw.relative_to(ROOT).as_posix()}',
        f'python3 scripts/format_metadata_host_results.py --raw {raw.relative_to(ROOT).as_posix()}',
        '```', '',
        'The collector sets OMP/MKL/Rayon/OpenBLAS/Julia thread counts to 1 via `configure_cpu_thread_env 1`, sets `CPU_FOLLOWUP_TARGET_NS=50000000`, and exports the installed MKL/compiler library paths. Host and container idle guards remain enabled. Exact process commands, timings and signatures are in the compressed records; [restore instructions](../../../mwe/cpu_followup/README.md) apply unchanged. Both typed arms and the existing Julia reference are registered under #2040 in `cpu/perf_issues` (manifest v5).']
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines)+'\n')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',type=Path,default=ROOT/'data/results/amd-cpu/cpu/metadata_host/20261008_host_views')
    parser.add_argument('--output',type=Path,default=ROOT/'result/amd-cpu/cpu/metadata_host.md')
    args=parser.parse_args()
    render(args.raw.resolve(),args.output.resolve())
