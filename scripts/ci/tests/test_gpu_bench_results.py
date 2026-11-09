"""One refusal test per mechanism (#420), week-number baseline selection, and the CLI.

A record is what an accepted run leaves behind: the summary document from
``result/<profile>/gpu/<suite>.summary.json`` (identity: ``tenferro_commit``,
``hardware.host``, ``timestamp``, measured rows). Two records that differ only
in one identity field must be refused by name and must not print a table.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.ci.gpu_bench_results import (  # noqa: E402
    accepted_runs, baseline_for, compare, render, status)

COMMIT_A = 'a' * 40
COMMIT_B = 'b' * 40
HOST = '941a9a84f64c'
# 2026-11-08 is a Sunday: ISO week 45.  2026-11-01 is week 44, 2026-10-25 week 43.
WEEK_45_LATE = '2026-11-08T11:52:37+00:00'
WEEK_45_EARLY = '2026-11-08T09:06:27+00:00'
WEEK_44 = '2026-11-01T11:00:00+00:00'
WEEK_43 = '2026-10-25T11:00:00+00:00'


def record(timestamp, commit, host=HOST, scale=1.0, unit='ms'):
    return {
        'suite_id': 'gpu/dense', 'target_profile': 'nvidia-gpu', 'tenferro_commit': commit,
        'timestamp': timestamp, 'unit': unit,
        'hardware': {'gpu': 'NVIDIA A100 80GB PCIe', 'host': host},
        'rows': [{'case': 'dense_matmul', 'backend': backend, 'median': scale, 'status': 'ok'}
                 for backend in ('tenferro-cuda-eager', 'pytorch-cuda')],
    }


class RefusalTests(unittest.TestCase):
    def assertRefused(self, result, clause):
        text = render(result)
        self.assertEqual(status(result), 1)
        self.assertIn(clause, text)
        self.assertNotIn('| case |', text)

    def test_identical_commit_is_refused_and_no_table_is_printed(self):
        # Only the collection time differs, so this pair also shares its week;
        # the commit clause is the mechanism under test.
        candidate = record(WEEK_45_LATE, COMMIT_B, scale=2.0)
        baseline = record(WEEK_45_EARLY, COMMIT_B)
        self.assertRefused(compare(candidate, baseline, 1.2), 'same tenferro_commit bbbbbbb')

    def test_different_host_is_refused_and_no_table_is_printed(self):
        # The host is the only mechanism: the commit and the week differ.
        candidate = record(WEEK_45_LATE, COMMIT_B, scale=2.0)
        baseline = record(WEEK_44, COMMIT_A, host='053e3b59f698')
        result = compare(candidate, baseline, 1.2)
        self.assertRefused(result, 'different host: 053e3b59f698 vs 941a9a84f64c')
        self.assertNotIn('same tenferro_commit', render(result))
        self.assertNotIn('same ISO week', render(result))

    def test_same_iso_week_is_refused_and_no_table_is_printed(self):
        # The week is the only mechanism: different commit, same host.
        candidate = record(WEEK_45_LATE, COMMIT_B, scale=2.0)
        baseline = record(WEEK_45_EARLY, COMMIT_A)
        result = compare(candidate, baseline, 1.2)
        self.assertRefused(result, 'same ISO week 2026-W45')
        self.assertNotIn('different host', render(result))
        self.assertNotIn('same tenferro_commit', render(result))


class BaselineTests(unittest.TestCase):
    def test_baseline_is_the_previous_weeks_accepted_run(self):
        candidate = record(WEEK_45_LATE, COMMIT_B, scale=1.5)
        same_week = record(WEEK_45_EARLY, COMMIT_A, scale=9.0)
        previous = record(WEEK_44, COMMIT_A)
        baseline = baseline_for([candidate, same_week, previous], candidate)
        self.assertEqual(baseline['timestamp'], WEEK_44)
        result = compare(candidate, baseline, 1.2)
        self.assertEqual(result['refusals'], [])
        self.assertEqual({row['ratio'] for row in result['rows']}, {1.5})
        self.assertEqual(status(result), 0)

    def test_the_newest_accepted_run_of_the_baseline_week_is_the_baseline(self):
        candidate = record(WEEK_45_LATE, COMMIT_B)
        early, late = record(WEEK_44, COMMIT_A, scale=1.0), None
        late = dict(early, timestamp='2026-11-01T23:00:00+00:00', tenferro_commit=COMMIT_B[:39] + 'c')
        self.assertEqual(baseline_for([candidate, late, early], candidate)['timestamp'],
                         '2026-11-01T23:00:00+00:00')

    def test_previous_week_without_an_accepted_run_says_no_baseline(self):
        # The neighbouring week is never substituted for the missing one.
        candidate = record(WEEK_45_LATE, COMMIT_B, scale=1.5)
        older = record(WEEK_43, COMMIT_A)
        baseline = baseline_for([candidate, older], candidate)
        self.assertIsNone(baseline)
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertIn('no baseline', text)
        self.assertIn('2026-W44', text)
        self.assertNotIn('| case |', text)
        self.assertEqual(status(result), 1)

    def test_report_states_both_runs_identity(self):
        candidate = record(WEEK_45_LATE, COMMIT_B, scale=1.5)
        baseline = record(WEEK_44, COMMIT_A)
        text = render(compare(candidate, baseline, 1.2))
        for fragment in (f'`{COMMIT_B[:7]}`', f'`{COMMIT_A[:7]}`', f'`{HOST}`',
                         '`2026-W45`', '`2026-W44`', 'pytorch-cuda, tenferro-cuda-eager'):
            self.assertIn(fragment, text)
        self.assertNotIn('refused', text)
        self.assertIn('| baseline ms | candidate ms |', text)

    def test_the_recorded_unit_labels_the_table(self):
        # gpu/perf_issues records ns/op, not ms.
        candidate = record(WEEK_45_LATE, COMMIT_B, scale=1.5, unit='ns/op')
        baseline = record(WEEK_44, COMMIT_A, unit='ns/op')
        self.assertIn('| baseline ns/op | candidate ns/op |',
                      render(compare(candidate, baseline, 1.2)))


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.git('init', '-q', '-b', 'main')
        self.git('config', 'user.email', 't@example.invalid')
        self.git('config', 'user.name', 't')

    def git(self, *args):
        subprocess.run(['git', '-C', str(self.repo), *args], check=True, capture_output=True)

    def accept(self, document):
        path = self.repo / f"result/{document['target_profile']}/{document['suite_id']}.summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))
        self.git('add', '-A')
        self.git('commit', '-q', '-m', 'Accept benchmark run 1 attempt 1')

    def run_tool(self):
        return subprocess.run(
            [sys.executable, str(ROOT / 'scripts/ci/gpu_bench_results.py'),
             '--repo', str(self.repo), '--profile', 'nvidia-gpu', '--suite', 'gpu/dense'],
            capture_output=True, text=True)

    def test_history_is_read_newest_first_from_the_summary_sidecar(self):
        self.accept(record(WEEK_44, COMMIT_A))
        self.accept(record(WEEK_45_LATE, COMMIT_B))
        records = accepted_runs(self.repo, 'nvidia-gpu', 'gpu/dense')
        self.assertEqual([item['timestamp'] for item in records], [WEEK_45_LATE, WEEK_44])

    def test_re_accepting_an_older_run_does_not_make_it_the_candidate(self):
        # The newer measurement is accepted first, the older one afterwards.
        self.accept(record(WEEK_45_LATE, COMMIT_B))
        self.accept(record(WEEK_44, COMMIT_A))
        records = accepted_runs(self.repo, 'nvidia-gpu', 'gpu/dense')
        self.assertEqual(records[0]['timestamp'], WEEK_45_LATE)

    def test_a_record_without_a_host_is_not_a_comparable_run(self):
        document = record(WEEK_44, COMMIT_A)
        del document['hardware']['host']
        self.accept(document)
        self.assertEqual(accepted_runs(self.repo, 'nvidia-gpu', 'gpu/dense'), [])

    def test_cli_prints_the_identity_then_the_table_and_exits_zero(self):
        self.accept(record(WEEK_44, COMMIT_A))
        self.accept(record(WEEK_45_LATE, COMMIT_B, scale=1.5))
        process = self.run_tool()
        self.assertEqual(process.returncode, 0)
        lines = process.stdout.splitlines()
        self.assertTrue(lines[0].startswith('# Accepted run comparison: gpu/dense'))
        self.assertIn(f'candidate: tenferro-rs `{COMMIT_B[:7]}`', lines[2])
        self.assertIn(f'baseline: tenferro-rs `{COMMIT_A[:7]}`', lines[3])
        self.assertIn('| dense_matmul | tenferro-cuda-eager | 1 | 1.5 | 1.50x |', process.stdout)

    def test_cli_refuses_on_stdout_and_exits_one_without_a_table(self):
        self.accept(record(WEEK_44, COMMIT_A))
        self.accept(record(WEEK_45_LATE, COMMIT_A, scale=1.5))
        process = self.run_tool()
        self.assertEqual(process.returncode, 1)
        self.assertIn(f"refused: same tenferro_commit {COMMIT_A[:7]}:", process.stdout)
        self.assertNotIn('| case |', process.stdout)


if __name__ == '__main__':
    unittest.main()
