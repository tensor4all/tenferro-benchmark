import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/results/amd-cpu/cpu/followup_12x'
CAT={c['id']:c for c in json.loads((ROOT/'mwe/cpu_followup/cases.json').read_text())}
PHASES=['20261008_confirmed_cpu','20261008_corrected_confirm','20261008_long_batches','20261008_materialize_views']

def final_rows():
 rows={}
 for phase in PHASES:
  for r in json.loads((RAW/phase/'confirm.json').read_text()):
   if phase=='20261008_confirmed_cpu' and (r['case_id'].startswith('fft.') or r['reference']=='julia-base'):continue
   rows[r['case_id']]=dict(r,campaign=phase)
 scans={}
 for phase in ['20261008_cpu_wheel','20261008_scatter_add','20261008_corrected_refs','20261008_cached_mkl_fft','20261008_long_batches','20261008_materialize_views']:
  for r in json.loads((RAW/phase/'scan.json').read_text()):
   if r.get('correct') and r.get('ratio') is not None:
    key=(r['case_id'],r['threads'],r['path']);scans[key]=dict(r,campaign=phase)
 for cid,c in CAT.items():
  if cid not in rows:
   candidates=[r for r in scans.values() if r['case_id']==cid]
   r=max(candidates,key=lambda r:r['ratio'])
   rows[cid]=dict(r,status='screened-below-threshold' if r['ratio']<1.2 else 'unconfirmed',tenferro_ns=r['rust']['median_ns'],reference_ns=r['ref']['median_ns'])
 for cid,r in rows.items():
  r['fixture']=CAT[cid]['fixture'];r['source_observations']=len(CAT[cid]['original_rows'])
  r['issue']=2021 if cid=='structural.broadcast_in_dim' else None
  if cid=='structural.broadcast_in_dim':r['disposition']='covered-by-open-issue'
  else:r['disposition']=r['status']
 return rows

def main():
 import collections
 rows=final_rows()
 posted=json.loads((RAW/'issue-reporting.json').read_text())
 for group in posted.values():
  for cid in group['cases']:rows[cid]['issue']=group['number']
 status=collections.Counter(r['disposition'] for r in rows.values())
 assert len(rows)==53
 assert sum(r['disposition']=='confirmed' for r in rows.values())==43
 output=ROOT/'result/amd-cpu/cpu/followup_12x.md'
 lines=['# CPU ≥1.2× follow-up: current public operation MWEs', '',
 '2026-10-08, AMD Ryzen 9 9955HX, Linux CPU MKL devcontainer. **43 operation workloads were confirmed and reported in 9 new tenferro-rs Issues.** One workload is already covered by open #2021. Two workloads remain inconclusive; seven did not remain ≥1.2× suspects in the operation-only 1T/4T scan. A scan below threshold is not a confirmed parity claim.', '',
 'Selection: 53 operation groups / 139 original condition rows from the earlier inventory without open owners, selected because at least one original ratio was ≥1.2. All 53 have runnable public-API reproducers. The earlier table rows are screening evidence only. Metadata fixtures are deliberately smaller; their original large-fixture timings are not confirmed. Broadcast #2021 was identified by a renewed full-body open-Issue search before filing and excluded from new reports.', '',
 'Library: `b3f47296244ff7b7c55ac0a75f782cb0835418c1`, pulled main then deliberately pinned for the campaign. tenferro BLAS linkage and native FFT reference: oneMKL 2026.0.1; tenferro FFT itself uses cached RustFFT 6.4.1 plans. PyTorch **2.12.0+cpu**, wheel MKL 2024.2. JAX/jaxlib **0.10.1**, CPU. Julia **1.13.1**. strided-rs **12ff2de906f4d894a4329f7dbf195b0f874f846e**. No CUDA, no runner taskset/numactl pinning; tenferro public backend-managed placement is retained. [Provider/config metadata](../../../data/results/amd-cpu/cpu/followup_12x/20261008_confirmed_cpu/run_t4.yaml) and [hardware](../../../data/results/amd-cpu/cpu/followup_12x/20261008_confirmed_cpu/hardware.json).', '',
 'Setup is untimed on every arm, including calibration: fixture creation/conversion, wrapping, contexts/sessions, input views/descriptors, FFT planning, JIT and first completed initialization. Timers contain only the declared operation, intrinsic allocation, output retention assignment and native completion. Results stay alive past the clock; downloads/signatures/validation/destruction are untimed. Direct/eager paths hold one entered session. Metadata is session-free. [MWE setup/commands](../../../mwe/cpu_followup/README.md), [timing audit](../../../mwe/cpu_followup/timing-audit.md).', '',
 'Confirmation: 3 warmups and 15 batched samples per process; four A/A pairs and four independent balanced comparison pairs AB/BA/BA/AB. Statistic: median of four round-median ratios. Declared gate: ratio ≥1.2, A/A median within 10%, every sample-set CoV ≤20%, numerical validation passed. The corrected/longer-batch phases also balance A/A and require each A/A round within 10%. No threshold was estimated from candidate results or loosened. Scan selects the strongest eligible 1T/4T condition; the table does not claim both thread counts are slow.', '',
 'Batch targets are 10 ms or 50 ms for longer-batch retries, bounded by 512 MiB retained tensor payloads/owned metadata inputs. One output is still retained for the 2 GiB rotation. The final 2×2 metadata-transpose reference uses ~1.86 million operations in ~1.5 ms at its memory cap; normalized sub-ns values are specialized bulk metadata throughput, not isolated call latency. Absolute tenferro metadata costs are hundreds of ns. Output shape/count, aggregates and ~130 deterministic probes agree; activation/permutation outputs additionally pass full scalar/odometer oracles. FFT checks input preservation as well.', '',
 'Compiled JAX uses dynamic arguments and completed execution. Its 1/4 active Eigen workers were verified with [untimed probes](../../../data/results/amd-cpu/cpu/followup_12x/20261008_confirmed_cpu/jax-runtime-t4.jsonl). Native layouts are preserved with equivalent logical values; no one-sided layout conversion is timed. FFT compares **cached native public oneMKL DFTI**, not PyTorch FFT: [PyTorch CPU FFT replans per call](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/mkl/SpectralOps.cpp#L490), so its historical rows are not operation-only evidence.', '',
 '## New reports', '', '| Issue | family | confirmed workloads | ratio range |', '|---|---|---:|---:|']
 for name,g in posted.items():
  rs=[rows[cid] for cid in g['cases']]
  lines.append(f"| [#{g['number']}]({g['url']}) | {name} | {len(rs)} | {min(r['ratio'] for r in rs):.3f}–{max(r['ratio'] for r in rs):.3f}× |")
 lines += ['', '## All selected workloads', '',
 'Times are per operation in µs. Confirmed rows show independent paired results; inconclusive ratios are diagnostic and were **not filed**. Screen-only rows show the strongest eligible scan condition. The exact physical permutation and source strides are in the [fixture definitions](../../../data/instances/permutation_patterns.json). The default MWE fixture catalog preserves original observation identities.', '',
 '| case ID | input shape | dtype | T | path | reference | tenferro µs | reference µs | ratio | disposition / owner | campaign |',
 '|---|---|---|---:|---|---|---:|---:|---:|---|---|']
 for cid,r in sorted(rows.items()):
  shape='×'.join(map(str,r['fixture']['input_shape']))
  owner=f"[#{r['issue']}](https://github.com/tensor4all/tenferro-rs/issues/{r['issue']})" if r['issue'] else '—'
  phase=r['campaign'];kind='confirm' if r['status'] not in ('screened-below-threshold','unconfirmed') else 'scan'
  lines.append(f"| `{cid}` | {shape} | {CAT[cid]['dtype']} | {r['threads']} | {r['path']} | {r['reference']} | {r['tenferro_ns']/1000:.6g} | {r['reference_ns']/1000:.6g} | {r['ratio']:.3f}× | {r['disposition']} / {owner} | [{phase}](../../../data/results/amd-cpu/cpu/followup_12x/{phase}/{kind}.json) |")
 lines += ['', 'The maintainer-requested pinned Host DynRank/static-rank extension for #2040 is reported separately in [CPU metadata Host comparisons](metadata_host.md); it remeasures all representations in balanced rounds and preserves the original TensorValue observations above.', '', '## Evidence and supersession', '',
 'The selected prepared-trace paths are explicitly **scope-ineligible**, since their public API cannot borrow an execution session and includes internal admission/session work. They are not counted as confirmed, parity or covered. This confirmation campaign uses the independent crate and dedicated registered-case runner. The root CPU-provider migration landed separately on main during the campaign; the original measurement revisions are preserved. The follow-up does not refresh the complete run_all suite.', '',
 'Canonical precedence is encoded in [the report formatter](../../../scripts/format_cpu_followup_results.py). Original confirmations involving Julia first-calibration JIT or PyTorch FFT planning are superseded, never used as operation evidence. The old transpose-metadata 32×32 case was inconclusive and is replaced by the 2×2 long batch. Earlier PyTorch materialization closures with timed input-view construction are superseded. Earlier scatter replacement semantics were corrected to additive scatter in both arms before eligible confirmation. Preliminary CUDA-wheel, constant-captured JAX, unspecialized Julia and incorrect Fortran-symbol DFTI probes are not publication evidence. No measured timing/status values were edited.', '',
 '| raw campaign | purpose and eligibility |', '|---|---|']
 for phase,purpose in [
  ('20261008_cpu_wheel','53-operation 1T/4T screen with CPU wheel; Julia/FFT/materialization subsets superseded below'),
  ('20261008_scatter_add','fair additive-scatter screen'),
  ('20261008_confirmed_cpu','first confirmation: use non-Julia/non-FFT cases, excluding materialization rows superseded below'),
  ('20261008_corrected_refs','untimed exact Julia batch JIT; metadata/reduction/complex-exp screen only, FFT probe failures excluded'),
  ('20261008_cached_mkl_fft','cached public native oneMKL FFT screen, input-preservation validation'),
  ('20261008_corrected_confirm','balanced A/A corrected Julia and FFT confirmation; transpose metadata superseded below'),
  ('20261008_long_batches','50 ms target and balanced A/A retry: metadata transpose/scatter/dynamic update confirmed; slogdet noisy'),
  ('20261008_materialize_views','untimed input-view preparation; diagonal/transpose confirmed; slice remains inconclusive; broadcast owned by #2021')]:
  lines.append(f'| [{phase}](../../../data/results/amd-cpu/cpu/followup_12x/{phase}/declaration.json) | {purpose} |')
 lines += ['', 'Per-process JSON is preserved losslessly in each campaign’s `processes.jsonl.gz` ([restore instructions](../../../mwe/cpu_followup/README.md)); it includes exact per-process commands, source revisions, counts, durations and validation signatures. [Submission identities](../../../data/results/amd-cpu/cpu/followup_12x/issue-reporting.json) retain MWE source commits and case membership. Workloads and reference arms are added to `cpu/perf_issues` at opening (manifest v4; Host extensions in v5); the exact existing activation cases are annotated instead of duplicated. Use `scripts/run_cpu_followup.sh` inside the Linux MKL devcontainer to run registered latest-main cases. No automated audit, registry, scheduled run or mandatory baseline campaign is introduced.']
 output.write_text('\n'.join(lines)+'\n')
 (RAW/'dispositions.json').write_text(json.dumps(list(rows.values()),indent=2)+'\n')
 print(json.dumps(dict(total=len(rows),dispositions=dict(status)),indent=2))

if __name__=='__main__':main()
