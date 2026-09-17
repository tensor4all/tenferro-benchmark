#!/usr/bin/env python3
import json
from pathlib import Path
root = Path(__file__).resolve().parent
report = {}
for directory in [root, root/'rerun']:
    records = json.loads((directory/'runs.json').read_text())
    activity = json.loads((directory/'activity.json').read_text())
    rows = {}
    for run in records:
        case,variant = run['case'],run['variant']
        assert run['exit_code'] == 0
        lines = [s.split() for s in (directory/f'{case}-{variant}.log').read_text().splitlines()
                 if s.startswith(case+' ')]
        assert len(lines)==2
        rows.setdefault(case,{})[variant] = {
            strategy:{'median_ms':float(v[4]), 'iqr_ms':float(v[5])}
            for strategy,v in zip(['opt_flops','opt_size'],lines)}
    known_foreign = [s for s in activity if any(
        line.split()[-1] in ['julia','pi'] for line in s['foreign_runnable'])]
    report[directory.name] = {'status':'INCONCLUSIVE' if known_foreign else 'REVIEW_REQUIRED',
        'known_foreign_samples':known_foreign,'results':rows}
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v['status'] for k,v in report.items()})
