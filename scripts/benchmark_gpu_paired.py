#!/usr/bin/env python3
"""Sequential CUDA A/A and declared baseline/candidate confirmation (#107)."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

import yaml
import bench_selection
import benchmark_perf_issues as suite
import case_status
import detect_regressions as detector
import gpu_routine_controls

ROOT = Path(__file__).resolve().parents[1]
FEATURES = 'native,cuda'


def output(args):
    return subprocess.check_output([str(a) for a in args], text=True).strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


@contextmanager
def environment(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build(checkout):
    checkout = checkout.resolve()
    if output(['git', '-C', checkout, 'branch', '--show-current']) == 'main':
        subprocess.run(['git', '-C', str(checkout), 'pull', '--ff-only'], check=True)
    revision = output(['git', '-C', checkout, 'rev-parse', 'HEAD'])
    if output(['git', '-C', checkout, 'status', '--porcelain', '--untracked-files=no']):
        raise ValueError('library checkout must be clean')
    link = ROOT / 'extern/tenferro-rs'
    previous = os.readlink(link) if link.is_symlink() else None
    if previous is None and link.resolve() != checkout:
        raise ValueError('cross-revision builds require a worktree with a symlinked extern/tenferro-rs')
    target = Path(os.environ.get('CARGO_TARGET_DIR', ROOT / 'target')) / 'gpu-paired' / revision[:12]
    try:
        if previous is not None:
            link.unlink()
            link.symlink_to(checkout, target_is_directory=True)
        with environment(dict(CARGO_TARGET_DIR=target, RUSTC_WRAPPER='', CUDA_PATH=os.environ.get('CUDA_HOME', '/usr/local/cuda'))):
            subprocess.run(['cargo', 'build', '--release', '--no-default-features', '--features', FEATURES,
                            '--bin', 'benchmark_gpu_perf_issues', '-j', '4'], cwd=ROOT, check=True)
    finally:
        if previous is not None:
            link.unlink()
            link.symlink_to(previous, target_is_directory=True)
    binary = target / 'release/benchmark_gpu_perf_issues'
    return dict(directory=str(checkout), revision=revision, binary=str(binary), binary_sha256=sha(binary),
                features=FEATURES, cargo_lock_sha256=sha(ROOT / 'Cargo.lock'))


def idle(device):
    apps = output(['nvidia-smi', '-i', device, '--query-compute-apps=pid,process_name', '--format=csv,noheader'])
    if apps:
        raise ValueError(f'physical GPU {device} is busy: {apps}')


def validate_config(config):
    if config['threads'] != [1] or config['build'] != dict(profile='release', features=FEATURES):
        raise ValueError('GPU confirmation requires release/native,cuda and threads=[1]')
    if config['cases']['suite_id'] != 'gpu/perf_issues' or not config['cases'].get('case_ids'):
        raise ValueError('declare the focused GPU case IDs explicitly')
    if 'status' in config:
        required_cuda = {'device_uuid','device_name','driver_version','runtime_version'}
        if config['host']['target_profile'] != 'nvidia-gpu' or set(config['host'].get('cuda') or {}) != required_cuda:
            raise ValueError('declare NVIDIA target profile and CUDA identity')
        for value in [*config['thresholds'].values(), config['noise']['max_cov'], config['noise']['max_aa_relative_spread']]:
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('thresholds/noise limits must be finite and positive')
    rounds = config['repetitions']['rounds']
    if not isinstance(rounds, int) or rounds < 2 or rounds % 2:
        raise ValueError('rounds must be positive, even and at least two')
    if config['repetitions']['runs'] < 3 or config['repetitions']['warmups'] < 0:
        raise ValueError('at least three samples and nonnegative warmups required')


def validate_campaign(paired, ids, reference, expected_commits):
    """Validate the declared subset without rewriting global suite coverage."""
    findings = []
    contracts = {}
    for entry in paired['plan']['order']:
        if entry['round'] not in paired['arms'].get(entry['arm'], {}):
            findings.append(dict(kind='MISSING', case=str(entry), detail='declared arm/round absent'))
    required = {case_status.case_key(key, 1) for key in ids}
    for arm, runs in paired['arms'].items():
        for rnd, run in runs.items():
            meta_path = Path(run['dir']) / 'run_t1.yaml'
            metadata = yaml.safe_load(meta_path.read_text())
            gpu = metadata['cuda']
            identity = {key: gpu.get(key) for key in reference}
            if identity != reference or sorted(metadata['tenferro_rs']['features']) != ['cuda', 'native']:
                findings.append(dict(kind='CHANGED_BASELINE', case=f'{arm}/{rnd}', detail='GPU identity/runtime/features differ'))
            if metadata['tenferro_rs']['commit'] != expected_commits[arm]:
                findings.append(dict(kind='CHANGED_BASELINE', case=f'{arm}/{rnd}', detail='library revision differs'))
            if metadata['harness']['commit'] != paired['plan']['harness_commit'] or metadata['harness'].get('dirty') or metadata['tenferro_rs'].get('dirty'):
                findings.append(dict(kind='CHANGED_BASELINE', case=f'{arm}/{rnd}', detail='dirty or different harness/library'))
            rows = run['rows']
            for key in sorted(required - rows.keys()):
                findings.append(dict(kind='MISSING', case=key, detail=f'{arm}/{rnd} declared case absent'))
            for key, row in rows.items():
                if key not in required:
                    findings.append(dict(kind='CHANGED_BASELINE', case=key, detail='unexpected case'))
                elif row.get('status') != 'passed' or not row['per_op'] or any(not math.isfinite(v) or v <= 0 for v in row['per_op']):
                    kind = 'NEWLY_UNSUPPORTED' if row.get('status') in {'unsupported','not_configured'} else ('LIVENESS_FAILURE' if row.get('status') == 'timeout' else 'NUMERICAL_FAILURE')
                    findings.append(dict(kind=kind, case=key, detail=f'{arm}/{rnd}: unsuccessful/unmeasured'))
                else:
                    if len(row['samples']) != paired['plan']['repetitions']['runs']:
                        findings.append(dict(kind='MISSING', case=key, detail='sample count differs'))
                    contract = {field: row.get(field) for field in ('kind','arm','dtype','params','measurement_scope','sequence','cache_mode','cache_reset_scope','operations_per_sequence')}
                    if key in contracts and contracts[key] != contract:
                        findings.append(dict(kind='CHANGED_BASELINE', case=key, detail='operation/cache/sequence contract differs'))
                    contracts[key] = contract
    return findings


def run(args):
    os.chdir(ROOT)
    if output(['git', 'status', '--porcelain', '--untracked-files=no', '--', '.', ':(exclude)result']):
        raise ValueError('use one committed, clean harness for both campaigns')
    harness = output(['git', 'rev-parse', 'HEAD'])
    config_bytes = args.config.read_bytes() if args.mode == 'confirm' else None
    config = bench_selection.load_confirmation_config(args.config) if config_bytes is not None else None
    if config_bytes is not None and args.config.read_bytes() != config_bytes:
        raise ValueError('declaration changed while loading')
    if config:
        validate_config(config)
        if config['harness_commit'] != harness or config['host']['hostname'] != socket.gethostname():
            raise ValueError('harness/hostname differs from declaration')
        ids = config['cases']['case_ids']
        reps = config['repetitions']
        coverage = config['cases']['coverage']
    else:
        ids = args.case_ids.split(',') if args.case_ids else gpu_routine_controls.plan('quick', 'standard')['timing']
        reps = dict(rounds=args.rounds, warmups=args.warmups, runs=args.runs)
        coverage = args.coverage
        validate_config(dict(threads=[1], build=dict(profile='release', features=FEATURES),
                             cases=dict(suite_id='gpu/perf_issues', case_ids=ids), repetitions=reps))
    with environment(dict(BENCH_COVERAGE=coverage, BENCH_INSTANCE=','.join(ids))):
        _, cases, manifest, _, _, selected, _ = suite.plan('gpu')
    if set(selected) != set(ids) or any(cases[key]['measurement_scope'] in {'counter_diagnostic', 'state_diagnostic'} for key in ids):
        raise ValueError('confirmation accepts selected latency cases only')
    if config and config['cases']['manifest_version'] != manifest['manifest_version']:
        raise ValueError('manifest differs from declaration')
    uuid = output(['nvidia-smi', '-i', args.device, '--query-gpu=uuid', '--format=csv,noheader'])
    if config and config['host']['cuda']['device_uuid'] != uuid:
        raise ValueError('GPU UUID differs from declaration')
    idle(args.device)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    root = ROOT / 'data/results/nvidia-gpu/gpu' / ('perf_issues_aa' if args.mode == 'aa' else 'perf_issues_paired') / stamp
    root.mkdir(parents=True, exist_ok=False)
    builds = {'a' if args.mode == 'aa' else 'baseline': build(args.baseline)}
    first = next(iter(builds.values()))
    if args.mode == 'aa':
        builds['b'] = first
    else:
        builds['candidate'] = first if args.candidate.resolve() == args.baseline.resolve() else build(args.candidate)
        for arm in builds:
            if builds[arm]['revision'] != config['library'][arm + '_commit']:
                raise ValueError(f'{arm} build differs from declaration')
        (root / 'confirmation.yaml').write_bytes(config_bytes)
    arms = list(builds)
    order = [dict(round=r, arm=arm, dir=f'r{r:02d}_{arm}') for r in range(1, reps['rounds']+1)
             for arm in (arms if r % 2 else arms[::-1])]
    plan = dict(order=order, case_ids=ids, manifest_version=manifest['manifest_version'], coverage=coverage,
                repetitions=reps, builds=builds, harness_commit=harness, uuid=uuid,
                config_sha256=hashlib.sha256(config_bytes).hexdigest() if config else None, command=shlex.join(sys.argv))
    write(root / 'plan.json', plan)
    common = dict(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, RAYON_NUM_THREADS=1,
                  MATMUL_NUM_THREADS=1, CUBECL_DEBUG_LOG=0, BENCH_INCLUDE_SETUP_DIAGNOSTICS=1,
                  BENCH_COVERAGE=coverage, BENCH_INSTANCE=','.join(ids), BENCH_EFFORT='aa' if config is None else 'confirm',
                  BENCH_AA_WARMUPS=reps['warmups'], BENCH_AA_RUNS=reps['runs'])
    if config:
        common['BENCH_CONFIRM_CONFIG'] = str(root/'confirmation.yaml')
    with environment(common):
        for entry in order:
            idle(args.device)
            if config and sha(root/'confirmation.yaml') != plan['config_sha256']:
                raise ValueError('frozen declaration changed')
            descriptor = builds[entry['arm']]
            if sha(descriptor['binary']) != descriptor['binary_sha256']:
                raise ValueError('built binary changed')
            folder = root / entry['dir']
            folder.mkdir()
            subprocess.run([sys.executable, str(ROOT/'scripts/collect_run_metadata.py'), '--suite-id', 'gpu/perf_issues',
                '--suite-file', 'benchmarks/gpu/perf_issues.yaml', '--target-profile', 'nvidia-gpu',
                '--timestamp', datetime.now(timezone.utc).isoformat(), '--tenferro-dir', descriptor['directory'],
                '--features', FEATURES, '--blas', 'none', '--cuda-device-ordinal', str(args.device),
                '--output', str(folder/'run_t1.yaml')], check=True)
            path = folder/'run_t1.yaml'
            metadata = yaml.safe_load(path.read_text())
            if config and {key: metadata['cuda'].get(key) for key in config['host']['cuda']} != config['host']['cuda']:
                raise ValueError('CUDA identity/runtime differs from declaration')
            metadata['collection'] = dict(manifest_version=manifest['manifest_version'], coverage=coverage,
                effort=common['BENCH_EFFORT'], declared_case_ids=ids, repetitions=reps, binary=descriptor)
            path.write_text(yaml.safe_dump(metadata, sort_keys=False))
            suite.collect(Path(descriptor['binary']), folder/'samples_t1.jsonl', 1, suite_key='gpu', device=0)
            idle(args.device)
    paired = detector.load_paired(root)
    meta = yaml.safe_load((root/order[0]['dir']/'run_t1.yaml').read_text())
    identity = {key: meta['cuda'].get(key) for key in ('device_uuid','device_name','driver_version','runtime_version')}
    findings = validate_campaign(paired, ids, identity, {arm: builds[arm]['revision'] for arm in arms})
    if config is None:
        detector.characterize_aa(paired, root/'aa.json', root/'report.md')
        verdicts = []
    else:
        aa = detector.load_paired(args.aa_dir)
        if (aa['plan']['case_ids'] != ids or aa['plan']['harness_commit'] != harness or aa['plan']['uuid'] != uuid
                or aa['plan']['repetitions'] != reps or aa['plan']['coverage'] != coverage
                or aa['plan']['manifest_version'] != manifest['manifest_version']):
            raise ValueError('A/A scope, harness or hardware differs')
        if aa['plan']['builds']['a']['binary_sha256'] != builds['baseline']['binary_sha256']:
            raise ValueError('A/A baseline binary differs')
        findings += validate_campaign(aa, ids, identity, {'a': builds['baseline']['revision'], 'b': builds['baseline']['revision']})
        comparable, verdicts = detector.confirm(config, paired, aa)
        findings += comparable
        result = dict(mode='confirm', status='DETERMINISTIC_FAILURE' if findings else ('REGRESSION' if any(v['verdict']=='REGRESSION' for v in verdicts) else 'OK'), deterministic=findings, timing=verdicts)
        write(root/'detector.json', result)
        (root/'report.md').write_text(detector.render(result))
    write(root/'findings.json', findings)
    prefix = (f'- Raw campaign: `{root.relative_to(ROOT)}`.\n\n- Command: `{plan["command"]}`.\n\n'
              'Focused declared subset only; globally unselected suite cases remain unmeasured. '
              'One declared unit is one allocation or one complete controlled shape sequence. '
              'Raw samples, sequence/cache contracts and build/hardware metadata remain in every arm.\n\n')
    latest = ROOT/'result/nvidia-gpu/gpu'/('perf_issues_aa.md' if config is None else 'perf_issues_paired.md')
    latest.write_text(prefix + (root/'report.md').read_text())
    print(root)
    return 2 if findings else int(any(v['verdict']=='REGRESSION' for v in verdicts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['aa','confirm'])
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--aa-dir', type=Path)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--runs', type=int, default=15)
    parser.add_argument('--coverage', choices=['quick','full'], default='quick')
    parser.add_argument('--case-ids')
    args = parser.parse_args()
    if args.mode == 'confirm' and not all((args.config,args.candidate,args.aa_dir)):
        parser.error('confirm requires --config, --candidate and --aa-dir')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
