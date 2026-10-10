"""Evaluate the predeclared complete CUDA concurrency comparison."""
from collections import Counter
from math import exp, log, sqrt
from statistics import mean, median, stdev


def evaluate(rows, protocol):
    if len(rows) != 10 or [r['arm'] for r in rows] != protocol['order']:
        raise ValueError('All ten samples in the declared order are required')
    if max(r['paid_seconds'] for r in rows[:2]) / min(r['paid_seconds'] for r in rows[:2]) > protocol['validity']['aa_max_min']:
        return {'verdict': 'INCONCLUSIVE', 'reason': 'A/A spread exceeded'}
    runs = rows[2:]
    a = [r for r in runs if r['arm']=='A']; b = [r for r in runs if r['arm']=='B']
    da=Counter(r['driver'] for r in a); db=Counter(r['driver'] for r in b)
    tv=sum(abs(da[k]/4-db[k]/4) for k in da.keys()|db.keys())/2
    ratios=[]
    for i in range(0,8,2):
        pair=runs[i:i+2]
        ratios.append(next(r['paid_seconds'] for r in pair if r['arm']=='B') / next(r['paid_seconds'] for r in pair if r['arm']=='A'))
    logs=[log(r) for r in ratios]
    radius=3.182446305284263*stdev(logs)/sqrt(4)  # two-sided t(3), 95%
    result={'mean_reduction':1-mean(r['paid_seconds'] for r in b)/mean(r['paid_seconds'] for r in a),
            'median_reduction':1-median(r['paid_seconds'] for r in b)/median(r['paid_seconds'] for r in a),
            'median_cuda_stage_reduction':1-median(r['stages']['cuda_tests']['seconds'] for r in b)/median(r['stages']['cuda_tests']['seconds'] for r in a),
            'candidate_max_over_baseline_max':max(r['paid_seconds'] for r in b)/max(r['paid_seconds'] for r in a),
            'driver_total_variation':tv,'pair_reductions':[1-r for r in ratios],
            'paired_geometric_mean_reduction':1-exp(mean(logs)),
            'paired_log_t_95_reduction_interval':[1-exp(mean(logs)+radius),1-exp(mean(logs)-radius)],
            'max_observed_gpu_memory_mib':max(r['gpu_memory_peak_mib'] for r in rows)}
    if tv>protocol['validity']['driver_patch_total_variation_between_arms_max']:
        return dict(result,verdict='INCONCLUSIVE',reason='Driver distribution imbalance')
    gate=protocol['acceptance']
    passed=(result['mean_reduction']>=gate['mean_paid_lifetime_reduction_min']
            and result['median_reduction']>=gate['median_paid_lifetime_reduction_min']
            and result['median_cuda_stage_reduction']>=gate['median_cuda_stage_reduction_min']
            and result['candidate_max_over_baseline_max']<=gate['max_candidate_paid_lifetime_over_max_baseline_max']
            and result['max_observed_gpu_memory_mib']<=protocol['validity']['max_observed_gpu_memory_mib'])
    return dict(result,verdict='PASS' if passed else 'FAIL')
