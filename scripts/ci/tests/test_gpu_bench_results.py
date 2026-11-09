"""One refusal test per mechanism (#420), week-number baseline selection, and the CLI.

A record is what an accepted run leaves behind: the summary document from
``result/<profile>/gpu/<suite>.summary.json`` (identity: ``tenferro_commit``,
``hardware.host``, ``timestamp``, measured rows). Two records that differ only
in one identity field must be refused by name and must not print a table.

Acceptance is a mark in the data, not the commit message: the ingestion driver
commits the sidecar together with ``<suite>.provenance.json``, whose ``files``
map holds the sidecar's sha256 (``result_ingestion.plan_updates``). A sidecar
revision without that receipt is an edit, not an accepted measurement, except
for the revisions from before receipts existed.
"""
import hashlib
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

    def test_a_unit_mismatch_is_refused_with_both_units(self):
        # Same duration, two units: the raw medians would divide to 1000000x and
        # the table would label the baseline with the candidate's unit.
        candidate = record(WEEK_45_LATE, COMMIT_B, scale=1000000, unit='ns/op')
        baseline = record(WEEK_44, COMMIT_A, unit='ms')
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertEqual(status(result), 1)
        self.assertIn('different unit: ms vs ns/op', text)
        self.assertNotIn('| case |', text)
        self.assertNotIn('1000000.00x', text)


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

    def write(self, relative, payload):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return relative

    def commit(self, message):
        self.git('add', '-A')
        self.git('commit', '-q', '-m', message)

    def sidecar(self, document):
        return f"result/{document['target_profile']}/{document['suite_id']}.summary.json"

    def receipt(self, relative, payload):
        return self.write(relative.replace('.summary.json', '.provenance.json'),
                          json.dumps({'files': {relative: hashlib.sha256(payload).hexdigest()}}).encode())

    def accept(self, document):
        """What the ingestion driver commits: the sidecar plus the receipt for its bytes."""
        payload = json.dumps(document).encode()
        relative = self.write(self.sidecar(document), payload)
        self.receipt(relative, payload)
        self.commit('Accept benchmark run 1 attempt 1')

    def edit(self, document, message='Unaccepted experiment'):
        """A sidecar edit with no acceptance: the committed receipt keeps the old digest."""
        self.write(self.sidecar(document), json.dumps(document).encode())
        self.commit(message)

    def run_tool(self, *args, repo=None):
        return subprocess.run(
            [sys.executable, str(ROOT / 'scripts/ci/gpu_bench_results.py'),
             '--repo', str(repo or self.repo), '--profile', 'nvidia-gpu', '--suite', 'gpu/dense', *args],
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

    def test_an_unaccepted_sidecar_edit_is_not_a_record(self):
        # A later collection time would make the edit the newest measurement if
        # every commit that touched the sidecar counted as accepted.
        self.accept(record(WEEK_44, COMMIT_A))
        self.accept(record(WEEK_45_EARLY, COMMIT_B))
        self.edit(record(WEEK_45_LATE, COMMIT_B, scale=9.0))
        records = accepted_runs(self.repo, 'nvidia-gpu', 'gpu/dense')
        self.assertEqual([item['timestamp'] for item in records], [WEEK_45_EARLY, WEEK_44])
        process = self.run_tool()
        self.assertEqual(process.returncode, 0)
        self.assertIn(f'candidate: tenferro-rs `{COMMIT_B[:7]}`', process.stdout)
        self.assertNotIn('9.00x', process.stdout)

    def test_a_new_suite_without_a_receipt_is_not_accepted(self):
        # The receipt mechanism exists (week 44 was ingested); a sidecar added
        # afterwards for a suite that was never ingested is not a record, and
        # stays out even when the driver later accepts that suite: the boundary
        # of the published history does not move with a suite's first receipt.
        self.accept(record(WEEK_44, COMMIT_A))
        edited = record(WEEK_45_EARLY, COMMIT_B)
        edited['suite_id'] = 'gpu/never_ingested'
        self.edit(edited)
        self.assertEqual(accepted_runs(self.repo, 'nvidia-gpu', 'gpu/never_ingested'), [])
        accepted = record(WEEK_45_LATE, COMMIT_B, scale=1.5)
        accepted['suite_id'] = 'gpu/never_ingested'
        self.accept(accepted)
        records = accepted_runs(self.repo, 'nvidia-gpu', 'gpu/never_ingested')
        self.assertEqual([item['timestamp'] for item in records], [WEEK_45_LATE])

    def test_a_sidecar_from_before_the_receipts_is_still_history(self):
        # Published before the driver recorded receipts: no receipt can exist.
        legacy = record(WEEK_44, COMMIT_A)
        self.write(self.sidecar(legacy), json.dumps(legacy).encode())
        self.commit('Write machine-readable report summaries')
        self.accept(record(WEEK_45_EARLY, COMMIT_B))
        records = accepted_runs(self.repo, 'nvidia-gpu', 'gpu/dense')
        self.assertEqual([item['timestamp'] for item in records], [WEEK_45_EARLY, WEEK_44])

    def test_a_later_receipt_accepts_an_already_edited_summary(self):
        # The driver may accept bytes that are already committed, changing only
        # the receipt, so the walk cannot stop at commits that touch the sidecar.
        self.accept(record(WEEK_44, COMMIT_A))
        self.edit(record(WEEK_45_LATE, COMMIT_B, scale=9.0), message='Manual sidecar edit')
        relative = f"result/nvidia-gpu/gpu/dense.summary.json"
        self.receipt(relative, (self.repo / relative).read_bytes())
        self.commit('Accept benchmark run 2 attempt 1')
        records = accepted_runs(self.repo, 'nvidia-gpu', 'gpu/dense')
        self.assertEqual([item['timestamp'] for item in records], [WEEK_45_LATE, WEEK_44])

    def test_deleting_the_receipt_does_not_accept_the_edit(self):
        self.accept(record(WEEK_44, COMMIT_A))
        self.accept(record(WEEK_45_EARLY, COMMIT_B))
        (self.repo / 'result/nvidia-gpu/gpu/dense.provenance.json').unlink()
        self.edit(record(WEEK_45_LATE, COMMIT_B, scale=9.0), message='Unaccepted experiment')
        records = accepted_runs(self.repo, 'nvidia-gpu', 'gpu/dense')
        self.assertEqual([item['timestamp'] for item in records], [WEEK_45_EARLY, WEEK_44])

    def test_a_receipt_free_suite_accepts_only_its_published_report_revisions(self):
        # A suite the driver never ingested: the run's report is published with
        # the sidecar, so a sidecar-only revision is an edit, not a run.
        for document in (record(WEEK_44, COMMIT_A), record(WEEK_45_EARLY, COMMIT_B)):
            relative = self.write(self.sidecar(document), json.dumps(document).encode())
            self.write(relative.replace('.summary.json', '.md'),
                       f"report of {document['timestamp']}\n".encode())
            self.commit('Publish the GPU page')
        self.edit(record(WEEK_45_LATE, COMMIT_B, scale=9.0))
        records = accepted_runs(self.repo, 'nvidia-gpu', 'gpu/dense')
        self.assertEqual([item['timestamp'] for item in records], [WEEK_45_EARLY, WEEK_44])

    def test_a_run_merged_from_a_branch_is_still_accepted(self):
        # Git's default path history simplification drops a commit whose sidecar
        # the merge did not keep (-s ours), so the week-44 run disappeared and the
        # tool said "no baseline" although the run is an ancestor of the candidate.
        self.accept(record(WEEK_43, COMMIT_A))
        self.git('checkout', '-q', '-b', 'feature')
        self.accept(record(WEEK_44, COMMIT_B[:39] + 'c'))
        self.git('checkout', '-q', 'main')
        self.accept(record(WEEK_45_LATE, COMMIT_B, scale=1.5))
        self.git('merge', '--no-ff', '-s', 'ours', 'feature', '-m', 'merge the feature branch')
        records = accepted_runs(self.repo, 'nvidia-gpu', 'gpu/dense')
        self.assertEqual([item['timestamp'] for item in records], [WEEK_45_LATE, WEEK_44, WEEK_43])
        process = self.run_tool()
        self.assertEqual(process.returncode, 0)
        self.assertIn('| baseline ms | candidate ms |', process.stdout)
        self.assertNotIn('no baseline', process.stdout)

    def test_cli_rejects_a_non_finite_or_out_of_range_threshold(self):
        self.accept(record(WEEK_44, COMMIT_A))
        self.accept(record(WEEK_45_LATE, COMMIT_B, scale=1.5))
        for value in ('nan', 'inf', '-inf', '0.5', '1000001', '1e308'):
            with self.subTest(threshold=value):
                process = self.run_tool(f'--threshold={value}')
                self.assertNotEqual(process.returncode, 0)
                self.assertEqual(process.stdout, '')
                self.assertIn('finite factor between 1 and 1000000', process.stderr)

    def test_cli_accepts_both_threshold_range_boundaries(self):
        self.accept(record(WEEK_44, COMMIT_A))
        self.accept(record(WEEK_45_LATE, COMMIT_B, scale=1.5))
        for value in ('1', '1000000'):
            with self.subTest(threshold=value):
                self.assertEqual(self.run_tool(f'--threshold={value}').returncode, 0)

    def test_a_git_failure_is_not_reported_as_missing_measurements(self):
        nowhere = Path(self.temp.name) / 'nonexistent'
        process = self.run_tool(repo=nowhere)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(process.stdout, '')
        self.assertIn(str(nowhere), process.stderr)
        self.assertNotIn('no accepted run', process.stdout + process.stderr)

    def test_a_shallow_checkout_is_refused_instead_of_reporting_no_baseline(self):
        # A truncated history cannot answer the week question either way.
        self.accept(record(WEEK_44, COMMIT_A))
        self.accept(record(WEEK_45_LATE, COMMIT_B, scale=1.5))
        shallow = Path(self.temp.name) / 'shallow'
        subprocess.run(['git', 'clone', '-q', '--depth', '1', f'file://{self.repo}', str(shallow)],
                       check=True, capture_output=True)
        process = self.run_tool(repo=shallow)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(process.stdout, '')
        self.assertIn('shallow', process.stderr)


if __name__ == '__main__':
    unittest.main()
