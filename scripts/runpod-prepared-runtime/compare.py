"""Evaluate the predeclared balanced cloud comparison without dropping samples."""
from collections import Counter
from itertools import combinations
from statistics import mean, median


def evaluate(rows, protocol):
    if len(rows) != 18 or [r['arm'] for r in rows] != protocol['order']:
        raise ValueError('The complete predeclared 18-run order is required')
    aa = rows[:2]
    if max(r['paid_seconds'] for r in aa) / min(r['paid_seconds'] for r in aa) > protocol['validity']['aa_max_min']:
        return {'verdict': 'INCONCLUSIVE', 'reason': 'A/A spread exceeded'}
    runs = rows[2:]
    a = [r for r in runs if r['arm'] == 'A']
    b = [r for r in runs if r['arm'] == 'B']
    drivers_a = Counter(r['driver'] for r in a)
    drivers_b = Counter(r['driver'] for r in b)
    tv = sum(abs(drivers_a[k] / 8 - drivers_b[k] / 8) for k in drivers_a.keys() | drivers_b.keys()) / 2
    mean_reduction = 1 - mean(r['paid_seconds'] for r in b) / mean(r['paid_seconds'] for r in a)
    median_reduction = 1 - median(r['paid_seconds'] for r in b) / median(r['paid_seconds'] for r in a)
    tail = max(r['paid_seconds'] for r in b) / max(r['paid_seconds'] for r in a)
    observed = mean(r['paid_seconds'] for r in a) - mean(r['paid_seconds'] for r in b)
    pairs = [runs[i:i + 2] for i in range(0, 16, 2)]
    permutations = []
    for candidate_first in combinations(range(8), 4):
        first = set(candidate_first)
        deltas = [pair[1 if i in first else 0]['paid_seconds'] - pair[0 if i in first else 1]['paid_seconds'] for i, pair in enumerate(pairs)]
        permutations.append(mean(deltas))
    p = sum(t >= observed - 1e-12 for t in permutations) / len(permutations)
    result = {'mean_reduction': mean_reduction, 'median_reduction': median_reduction,
              'candidate_max_over_baseline_max': tail, 'driver_total_variation': tv,
              'randomization_p': p, 'randomization_assignments': len(permutations),
              'pair_reductions': [1 - next(r['paid_seconds'] for r in pair if r['arm'] == 'B') / next(r['paid_seconds'] for r in pair if r['arm'] == 'A') for pair in pairs]}
    if tv > protocol['validity']['driver_patch_total_variation_between_arms_max']:
        return dict(result, verdict='INCONCLUSIVE', reason='Driver distribution imbalance')
    gate = protocol['acceptance']
    passed = (mean_reduction >= gate['mean_paid_lifetime_reduction_min']
              and median_reduction >= gate['median_paid_lifetime_reduction_min']
              and p <= gate['one_sided_balanced_block_randomization_p_max']
              and tail <= gate['max_candidate_paid_lifetime_over_max_baseline_max'])
    return dict(result, verdict='PASS' if passed else 'FAIL')
