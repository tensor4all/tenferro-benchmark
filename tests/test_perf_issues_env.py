"""Non-timing integration: missing MKL must not fail the CPU orchestrator."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark_perf_issues as suite


class MissingMklIntegrationTest(unittest.TestCase):
    def test_run_all_exits_zero_with_eight_skipped_fft_rows_per_thread(self):
        cases = json.loads((ROOT / 'data/instances/perf_issues.json').read_text())
        fft = [c['id'] for c in cases if c['kind'] == 'cpu_followup_mwe'
               and c['params']['case_id'].startswith('fft.')]
        self.assertEqual(len(fft), 8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Run the real shell orchestrators, collector, reference process,
            # accounting and renderer. Build/metadata/unrelated suites are fixtures.
            for directory in ('scripts', 'schemas', 'benchmarks', 'data/instances', 'mwe/cpu_followup'):
                shutil.copytree(ROOT / directory, root / directory,
                                ignore=shutil.ignore_patterns('__pycache__', 'target'))
            (root / '.venv').symlink_to(sys.prefix, target_is_directory=True)
            provider = root / 'scripts/cpu_blas_provider.sh'
            provider.write_text('if [[ "${RUN_ALL_MAIN_ONLY:-0}" == 1 ]]; then exit 0; fi\n'
                                + provider.read_text())
            (root / 'scripts/collect_run_metadata.py').write_text(
                "import pathlib,sys,yaml\n"
                "p=pathlib.Path(sys.argv[sys.argv.index('--output')+1])\n"
                "p.write_text(yaml.safe_dump(dict(schema_version=1,suite_id='cpu/perf_issues',"
                "target_profile='amd-cpu',timestamp='fixture',environment={})))\n")
            bin_dir = root / 'bin'
            bin_dir.mkdir()
            cargo = bin_dir / 'cargo'
            cargo.write_text('#!/bin/sh\n# No build, and no build-time MKLROOT.\n'
                             'test -z "${MKLROOT:-}"\n')
            cargo.chmod(0o755)
            hooks = root / 'hooks'
            hooks.mkdir()
            (hooks / 'sitecustomize.py').write_text(
                "import ctypes,ctypes.util\n"
                "original=ctypes.CDLL\n"
                "def unavailable(name,*args,**kwargs):\n"
                "    if 'libmkl_rt.so' in str(name):\n"
                "        raise OSError('test fixture: reference runtime absent')\n"
                "    return original(name,*args,**kwargs)\n"
                "ctypes.CDLL=unavailable\n"
                "ctypes.util.find_library=lambda name: None\n")
            env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
                       PYTHONPATH=str(hooks), BENCHMARK_COMMIT='fixture',
                       BENCHMARK_TIMESTAMP='skip-check', SKIP_EXTERN_SETUP='1',
                       TENFERRO_CPU_FEATURES='native', BENCH_COVERAGE='full',
                       BENCH_INSTANCE=','.join(fft), PERF_ISSUES_CORRECTNESS_ONLY='1',
                       RUN_FFT_SUITE='0', RUN_PUBLIC_API_SUITE='0', RUN_SMALL_WORK_SUITE='0',
                       RUN_OVERHEAD_SMOKE_SUITE='0', RUN_PERMUTATION_SUITE='0',
                       RUN_ELEMENTWISE_SUITE='0', RUN_PERF_ISSUES_SUITE='1')
            for key in ('MKLROOT', 'RUN_ALL_MAIN_ONLY', 'BENCHMARK_RAW_ONLY'):
                env.pop(key, None)
            result = subprocess.run(['bash', 'scripts/run_all.sh', '1', '4', '8'],
                                    cwd=root, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run = root / 'data/results/amd-cpu/cpu/perf_issues/skip-check'
            for threads in (1, 4, 8):
                line = f'Case status t{threads}: executed=0 failed=0 skipped=8 missing=0'
                self.assertIn(line, result.stdout)
                print(line)
                rows = [json.loads(s) for s in (run / f'samples_t{threads}.jsonl').read_text().splitlines()]
                self.assertTrue(all(r['status'] == 'skipped' and not r['samples'] for r in rows))
                self.assertTrue(all('test fixture: reference runtime absent' in r['skip_reason'] for r in rows))
            summary = json.loads((run / 'perf_issues.summary.json').read_text())
            import jsonschema
            jsonschema.validate(summary, json.loads((ROOT / 'schemas/report-summary.schema.json').read_text()))
            published = json.loads((root / 'result/amd-cpu/cpu/perf_issues.summary.json').read_text())
            self.assertEqual(published['counts'], summary['counts'])
            self.assertEqual(summary['counts']['skipped'], 24)
            self.assertEqual(summary['counts']['failed'], 0)
            self.assertEqual(summary['counts']['missing'], 0)
            self.assertTrue(all(r['status'] == 'skipped' and r['reason'] for r in summary['rows']))
            report = (run / 'report.md').read_text()
            self.assertIn('| expected | selected | executed | unsupported | failed | skipped |', report)
            self.assertIn('failed 0, skipped 24, missing 0', report)
            self.assertIn('SKIPPED:', report)
            self.assertFalse(json.loads((run / 'case_status.json').read_text())['complete'])
            print(f'run_all.sh 1 4 8 exit={result.returncode}; summary: failed=0 skipped=24 missing=0')

            # A real reference failure still fails the shell, but only after
            # rows, case status, report and summary have been written.
            hook = hooks / 'sitecustomize.py'
            hook.write_text(hook.read_text().replace('raise OSError(', 'raise RuntimeError('))
            env['BENCHMARK_TIMESTAMP'] = 'failure-check'
            failed = subprocess.run(['bash', 'scripts/run_all.sh', '1', '4'],
                                    cwd=root, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(failed.returncode, 1, failed.stdout + failed.stderr)
            failed_run = run.parent / 'failure-check'
            failed_summary = json.loads((failed_run / 'perf_issues.summary.json').read_text())
            self.assertEqual(failed_summary['counts']['failed'], 16)
            self.assertEqual(failed_summary['counts']['skipped'], 0)
            self.assertTrue((failed_run / 'report.md').is_file())
            self.assertTrue((failed_run / 'case_status.json').is_file())
            self.assertTrue((failed_run / 'samples_t1.jsonl').is_file())
            self.assertTrue((failed_run / 'samples_t4.jsonl').is_file())
            print(f'run_all.sh 1 4 exit={failed.returncode}; summary: failed=16 skipped=0; '
                  'raw/status/report/summary preserved')

    def test_genuine_execution_error_still_returns_nonzero(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                'BENCH_COVERAGE': 'full', 'BENCH_INSTANCE': 'linalg_lane_plan_counter'}):
            binary = Path(tmp) / 'broken'
            binary.write_text('#!/bin/sh\necho "execution failed" >&2\nexit 1\n')
            binary.chmod(0o755)
            self.assertTrue(suite.collect(binary, Path(tmp) / 'samples_t1.jsonl', 1, True))
            status = json.loads((Path(tmp) / 'case_status_t1.json').read_text())
            self.assertEqual(status['counts']['failed'], 1)
            self.assertEqual(status['counts']['skipped'], 0)


if __name__ == '__main__':
    unittest.main()
