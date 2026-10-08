"""Evaluate the predeclared complete-workload, three-pair paid GPU time campaign."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def compare(samples: list[dict]) -> dict:
    order = ['baseline', 'candidate', 'candidate', 'baseline', 'baseline', 'candidate']
    reasons = []
    if [s.get('arm') for s in samples] != order:
        return {'verdict': 'INCONCLUSIVE', 'reasons': ['Expected all six runs in the predeclared pair order']}
    if len({s.get('run_id') for s in samples}) != 6:
        reasons.append('Runs must be distinct')
    for arm in ('baseline', 'candidate'):
        refs = {s.get('tested_ref') for s in samples if s.get('arm') == arm}
        if len(refs) != 1 or not next(iter(refs), None):
            reasons.append(f'{arm} must execute one frozen source ref')
    costs = []
    for sample in samples:
        if sample.get('phase') != 'confirmation':
            reasons.append('Diagnostic pilots cannot enter confirmation')
        if (sample.get('cuda_passed'), sample.get('pjrt_passed'), sample.get('tutorial_passed'),
            sample.get('cleanup_confirmed'), sample.get('gpu_job_conclusion')) != (285, 3, True, True, 'success'):
            reasons.append(f"Run {sample.get('run_id')} lacks a complete successful workload and cleanup")
        seconds, price = sample.get('paid_seconds'), sample.get('price_per_hour')
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0
               for v in (seconds, price)):
            reasons.append('Missing or invalid actual paid duration/price')
            continue
        costs.append(seconds * price / 3600)
    if reasons:
        return {'verdict': 'INCONCLUSIVE', 'reasons': reasons}
    durations = [sample['paid_seconds'] for sample in samples]
    by_arm = {arm: [seconds for sample, seconds in zip(samples, durations) if sample['arm'] == arm]
              for arm in ('baseline', 'candidate')}
    for arm, values in by_arm.items():
        if max(values) / min(values) > 1.5:
            reasons.append(f'{arm} max/min paid seconds exceeds 1.5')
    pairs = [{samples[i]['arm']: durations[i], samples[i + 1]['arm']: durations[i + 1]} for i in range(0, 6, 2)]
    baseline = statistics.median(by_arm['baseline'])
    candidate = statistics.median(by_arm['candidate'])
    reduction = 1 - candidate / baseline
    verdict = 'INCONCLUSIVE' if reasons else 'PASS' if reduction >= 0.2 and all(
        pair['candidate'] <= pair['baseline'] for pair in pairs) else 'FAIL'
    return {'verdict': verdict, 'reasons': reasons, 'baseline_median_paid_seconds': baseline,
            'candidate_median_paid_seconds': candidate,
            'baseline_median_cost': statistics.median([cost for sample, cost in zip(samples, costs) if sample['arm'] == 'baseline']),
            'candidate_median_cost': statistics.median([cost for sample, cost in zip(samples, costs) if sample['arm'] == 'candidate']), 'reduction_fraction': reduction,
            'pair_paid_seconds': pairs, 'samples': samples}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.samples.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}, indent=2))
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
