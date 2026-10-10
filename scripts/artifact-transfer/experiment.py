"""CPU-hosted transport experiment; no numerical/GPU execution."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'result/linux-cpu/ci/artifact-transfer-protocol.json'
CASES = {'common': 'runtime', 'cuda12.8': 'sdk'}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def prepare():
    manifest = {}
    for case, stem in CASES.items():
        source = Path('source') / case
        expected = (source / f'{stem}.sha256').read_text().split()[0]
        combined = hashlib.sha256()
        parts = []
        for i in range(5):
            name = f'{stem}.part{i:02}'
            path = source / name
            with path.open('rb') as f:
                while block := f.read(1024 * 1024):
                    combined.update(block)
            parts.append({'name': name, 'size': path.stat().st_size, 'sha256': digest(path)})
            zipped = Path('upload/zip') / case / f'{i:02}'
            zipped.mkdir(parents=True)
            os.link(path, zipped / name)
            if i == 0:
                shutil.copyfile(source / f'{stem}.sha256', zipped / f'{stem}.sha256')
            raw = Path('upload/raw'); raw.mkdir(exist_ok=True)
            os.link(path, raw / f'raw-{case}-part{i:02}')
        assert combined.hexdigest() == expected, f'{case}: source checksum mismatch'
        manifest[case] = {'sha256': expected, 'parts': parts}
    Path('manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def begin(index, case):
    path = Path('received'); shutil.rmtree(path, ignore_errors=True)
    path.mkdir()
    Path('sample-start.json').write_text(json.dumps({
        'index': int(index), 'case': case, 'start': time.monotonic(),
        'load_before': os.getloadavg(),
    }))


def end(arm):
    stop = time.monotonic()
    row = json.loads(Path('sample-start.json').read_text())
    row.update(arm=arm, seconds=stop-row.pop('start'), load_after=os.getloadavg())
    case = row['case']; stem = CASES[case]
    expected = json.loads(Path('manifest.json').read_text())[case]
    combined = hashlib.sha256()
    for i, part in enumerate(expected['parts']):
        path = Path('received') / (part['name'] if arm == 'A' else f'raw-{case}-part{i:02}')
        assert path.stat().st_size == part['size'] and digest(path) == part['sha256'], str(path)
        with path.open('rb') as f:
            while block := f.read(1024*1024):
                combined.update(block)
    assert combined.hexdigest() == expected['sha256'], case
    if arm == 'A':
        assert (Path('received') / f'{stem}.sha256').read_text().split()[0] == expected['sha256']
    row['verified'] = True
    with Path('observations.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row))


def evaluate(rows, protocol):
    from statistics import mean, median
    expected = [(i, c, a) for i, a in enumerate(protocol['order']) for c in CASES]
    if [(r['index'], r['case'], r['arm']) for r in rows] != expected or not all(r['verified'] for r in rows):
        return {'verdict': 'INCONCLUSIVE', 'reason': 'missing/order/identity mismatch'}
    aa = {c: max(r['seconds'] for r in rows if r['index'] < 2 and r['case']==c) /
          min(r['seconds'] for r in rows if r['index'] < 2 and r['case']==c) for c in CASES}
    totals = [sum(r['seconds'] for r in rows if r['index']==i) for i in range(10)]
    a = [totals[i] for i in range(2, 10) if protocol['order'][i]=='A']
    b = [totals[i] for i in range(2, 10) if protocol['order'][i]=='B']
    ratios = {c: median(r['seconds'] for r in rows if r['index']>=2 and r['case']==c and r['arm']=='B') /
              median(r['seconds'] for r in rows if r['index']>=2 and r['case']==c and r['arm']=='A') for c in CASES}
    result = {'aa_ratios': aa, 'mean_reduction': 1-mean(b)/mean(a), 'median_reduction': 1-median(b)/median(a),
              'case_median_ratios': ratios, 'baseline_seconds': a, 'candidate_seconds': b}
    if max(aa.values()) > protocol['aa_max_min']:
        return dict(result, verdict='INCONCLUSIVE', reason='A/A variability exceeded')
    passed = (result['mean_reduction'] >= protocol['min_reduction'] and
              result['median_reduction'] >= protocol['min_reduction'] and max(ratios.values()) <= protocol['per_case_max_median_ratio'])
    return dict(result, verdict='PASS' if passed else 'FAIL')


if __name__ == '__main__':
    command, *args = sys.argv[1:]
    if command == 'prepare': prepare()
    elif command == 'begin': begin(*args)
    elif command == 'end': end(*args)
    elif command == 'evaluate':
        rows = [json.loads(x) for x in Path('observations.jsonl').read_text().splitlines()]
        result = evaluate(rows, json.loads(PROTOCOL.read_text()))
        Path('decision.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result))
    else: raise SystemExit('Unknown command')
