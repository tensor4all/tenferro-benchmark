import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts.ci.gpu_readback import case_ids, validate


class ReadbackTests(unittest.TestCase):
    def test_selection_includes_all_sizes_operations_and_references(self):
        rows = json.loads(Path('data/instances/gpu_perf_issues.json').read_text())
        selected = [r for r in rows if r['id'] in case_ids()]
        self.assertEqual(len(selected), 16)
        self.assertEqual({(r['params']['n'], r['params']['op'], r['backend']) for r in selected},
                         {(n, op, backend) for n in (2, 4, 8, 32) for op in ('solve', 'qr')
                          for backend in ('tenferro-cuda', 'pytorch-cuda')})

    def test_incomplete_failed_or_unmeasured_rows_fail(self):
        row = dict(case_id='a', correctness_status='passed', regime='latency',
                   samples=[dict(iterations=8, elapsed_ns=1200)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'samples.jsonl'
            def check(rows, verify=False):
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                validate(path, ['a'], verify)
            check([row])
            check([dict(row, samples=[])], True)
            for rows in ([], [row, row], [dict(row, case_id='other')],
                         [dict(row, correctness_status='unsupported')],
                         [dict(row, regime='throughput')], [dict(row, samples=[])],
                         [dict(row, samples=[dict(iterations=0, elapsed_ns=1200)])],
                         [dict(row, samples=[dict(iterations=8, elapsed_ns=float('nan'))])]):
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    check(rows)
            with self.assertRaises(ValueError):
                check([row], True)

    def test_correctness_gate_stops_before_timing(self):
        source = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ('scripts/ci/run_gpu_readback.sh', 'scripts/ci/gpu_readback.py',
                             'data/instances/gpu_perf_issues.json'):
                dest = root / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, dest)
            smi = root / 'nvidia-smi'
            smi.write_text('#!/bin/sh\necho NVIDIA A100 80GB PCIe\n')
            smi.chmod(0o755)
            runner = root / 'scripts/run_gpu_perf_issues.sh'
            runner.write_text('''#!/usr/bin/env python3
import json, os
from pathlib import Path
verify = os.environ['PERF_ISSUES_CORRECTNESS_ONLY'] == '1'
with open('calls', 'a') as f: f.write('verify\\n' if verify else 'timing\\n')
out = Path('data/results/nvidia-gpu/gpu/perf_issues') / os.environ['BENCHMARK_TIMESTAMP']
out.mkdir(parents=True)
rows = [dict(case_id=cid, regime='latency', correctness_status='passed',
             samples=[] if verify else [dict(iterations=8,elapsed_ns=1200)])
        for cid in os.environ['BENCH_INSTANCE'].split(',')]
if os.environ.get('FAIL_VERIFY') == '1' and verify:
    rows[0]['correctness_status'] = 'failed'
(out/'samples_t1.jsonl').write_text(''.join(json.dumps(r)+'\\n' for r in rows))
''')
            runner.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + ':' + os.environ['PATH'], FAIL_VERIFY='1')
            result = subprocess.run(['bash', str(root/'scripts/ci/run_gpu_readback.sh')],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((root/'calls').read_text(), 'verify\n')
            (root/'calls').unlink()
            shutil.rmtree(root/'data/results')
            env.pop('FAIL_VERIFY')
            result = subprocess.run(['bash', str(root/'scripts/ci/run_gpu_readback.sh')],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root/'calls').read_text(), 'verify\ntiming\n')
