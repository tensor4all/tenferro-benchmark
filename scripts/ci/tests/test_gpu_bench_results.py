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
import os
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
        'rows': [{'case': 'dense_matmul', 'group': 'matmul', 'backend': backend, 'median': scale,
                  'status': 'ok', 'verified': True}
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

    def test_rows_that_differ_only_in_group_are_distinct(self):
        # The ingestion identity is (case, group, backend); keying on (case,
        # backend) lets g2 overwrite g1 and the g1 regression disappears.
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        baseline['rows'] = [dict(baseline['rows'][1], group='g1', median=1.0),
                            dict(baseline['rows'][1], group='g2', median=100.0)]
        candidate['rows'] = [dict(candidate['rows'][1], group='g1', median=10.0),
                             dict(candidate['rows'][1], group='g2', median=100.0)]
        result = compare(candidate, baseline, 1.2)
        self.assertEqual([(row['case'], row['group'], row['backend'], row['ratio'])
                          for row in result['rows']],
                         [('dense_matmul', 'g1', 'pytorch-cuda', 10.0)])
        self.assertIn('| dense_matmul | g1 | pytorch-cuda | 1 | 10 | 10.00x |', render(result))

    def test_a_zero_baseline_ratio_is_reported_not_dropped(self):
        # A zero median is a legal measurement (ingestion rejects only value < 0),
        # so 0 -> positive must be reported instead of vanishing as "(none)".
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        baseline['rows'][0]['median'] = 0.0
        candidate['rows'][0]['median'] = 10.0
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertEqual(result['rows'], [])
        self.assertIn('Ratio undefined (the baseline median is 0) for 1 row(s):', text)
        self.assertIn('| dense_matmul | matmul | tenferro-cuda-eager | 0 | 10 |', text)

    def test_a_duplicate_identity_is_refused_instead_of_overwritten(self):
        # result_ingestion rejects a summary with a repeated identity, so the
        # comparison must not keep the last of the two rows: a 10 -> 1 pair then
        # disappears and the table says "(none)".
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate['rows'] = [dict(candidate['rows'][0], median=10.0),
                             dict(candidate['rows'][0], median=1.0)]
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertEqual(result['rows'], [])
        self.assertNotIn('| case |', text)
        self.assertIn('candidate dense_matmul/matmul/tenferro-cuda-eager: duplicate identity', text)

    def test_a_negative_iqr_row_is_not_a_regression(self):
        # result_ingestion rejects a negative or non-finite IQR, so a receipt-free
        # record carrying one is not a 10x regression to report.
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate['rows'][0].update(median=10.0, iqr=-1)
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertEqual(result['rows'], [])
        self.assertNotIn('10.00x', text)
        self.assertIn('candidate dense_matmul/matmul/tenferro-cuda-eager: IQR is not finite and '
                      'non-negative', text)

    def test_an_unverified_row_without_a_counterpart_is_still_named(self):
        # Verification was inspected only after intersecting the measured
        # identities, so a row without a counterpart vanished without a word.
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate['rows'] = [{'case': 'new', 'group': 'matmul', 'backend': 'tenferro-cuda-eager',
                              'median': 1.0, 'status': 'ok', 'verified': False}]
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertEqual([row['case'] for row in result['unverified']], ['new'])
        self.assertIn('Skipped 1 unverified row(s)', text)
        self.assertIn('new/matmul/tenferro-cuda-eager (candidate unverified)', text)

    def test_a_row_whose_counterpart_is_not_ok_is_named(self):
        # The identity exists on both sides but the candidate recorded a skip, so
        # no ratio exists; the row used to disappear without a diagnostic.
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate['rows'][0] = {'case': 'dense_matmul', 'group': 'matmul',
                                'backend': 'tenferro-cuda-eager', 'status': 'skipped',
                                'reason': 'no kernel', 'median': None, 'iqr': None}
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertEqual(result['rows'], [])
        self.assertIn('dense_matmul/matmul/tenferro-cuda-eager (candidate status skipped)', text)

    def test_an_unhashable_group_is_refused_without_crashing(self):
        # The identity was hashed before its type was checked, so a list group
        # raised TypeError instead of naming the row.
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate["rows"][0]["group"] = []
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertEqual(result["rows"], [])
        self.assertIn("group is not a string", text)
        self.assertEqual(status(result), 1)

    def test_a_backend_the_renderer_cannot_sort_is_still_named(self):
        # The refusal is computed, but the arm set was sorted before it printed.
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate["rows"][0]["backend"] = None
        text = render(compare(candidate, baseline, 1.2))
        self.assertIn("backend is not a non-empty string", text)
        self.assertIn("invalid: candidate", text)

    def test_a_verification_mode_outside_the_schema_is_refused(self):
        # The schema enumerates the modes; anything else is a row the driver
        # never accepted, so it cannot be a regression to report.
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate["rows"][0].update(median=10.0, verification="made_up")
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertEqual(result["rows"], [])
        self.assertNotIn("10.00x", text)
        self.assertIn("verification made_up is not a recorded mode", text)
        self.assertEqual(status(result), 1)

    def test_a_record_without_a_row_array_is_refused_and_printed(self):
        # The refusal was computed, but render() walked the rows to state the arm
        # set first, so the CLI died before printing it.
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate["rows"] = None
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertIn("candidate rows is not an array", text)
        self.assertEqual(status(result), 1)

    def test_an_explicit_null_verification_is_refused(self):
        # The schema allows an absent verification mode, not a null one, so the
        # two cases cannot share the same guard.
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate["rows"][0].update(median=10.0, verification=None)
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertEqual(result["rows"], [])
        self.assertNotIn("10.00x", text)
        self.assertIn("verification None is not a recorded mode", text)
        self.assertEqual(status(result), 1)

    def test_a_status_the_driver_does_not_record_is_refused(self):
        # The non-ok branch accepted any nonempty status, so a typo status read
        # as a skipped row instead of the record the driver rejects.
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate["rows"][0] = {"case": "dense_matmul", "group": "matmul",
                                "backend": "tenferro-cuda-eager", "status": "typo",
                                "median": None, "verified": False}
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertNotIn("| case |", text)
        self.assertIn("status typo is not a status the driver records", text)
        self.assertEqual(status(result), 1)

    def test_a_skipped_row_still_obeys_the_schema_fields(self):
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate["rows"][0] = {"case": "dense_matmul", "group": "matmul",
                                "backend": "tenferro-cuda-eager", "status": "skipped",
                                "reason": "no kernel", "median": None, "iqr": None,
                                "verified": "yes", "verification": "made_up"}
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertIn("verified is not a boolean", text)
        self.assertIn("verification made_up is not a recorded mode", text)
        self.assertEqual(status(result), 1)

    def test_a_measure_only_row_cannot_be_verified(self):
        # The schema requires verified=false for measure_only and the driver
        # validates that schema, so this row is refused instead of compared.
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate["rows"][0].update(median=10.0, verification="measure_only", reason="not checked")
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertEqual(result["rows"], [])
        self.assertNotIn("10.00x", text)
        self.assertIn("candidate dense_matmul/matmul/tenferro-cuda-eager: measure_only row marked verified", text)

    def test_an_integer_too_large_for_a_float_is_refused_by_name(self):
        # JSON puts no bound on an integer, and math.isfinite raises
        # OverflowError on 10**400 instead of returning False.
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate["rows"][0]["median"] = 10 ** 400
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertEqual(result["rows"], [])
        self.assertIn("candidate dense_matmul/matmul/tenferro-cuda-eager: median is not a finite "
                      "non-negative number", text)

    def test_a_non_string_group_is_refused_instead_of_crashing_the_sort(self):
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate["rows"][0]["group"] = None
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertEqual(result["rows"], [])
        self.assertIn("dense_matmul/None/tenferro-cuda-eager: group is not a string", text)
        self.assertEqual(status(result), 1)

    def test_a_case_that_disappears_from_the_candidate_is_named(self):
        # A baseline row the candidate run did not record at all used to leave
        # the table empty with no diagnostic: lost coverage read as no change.
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate["rows"] = candidate["rows"][1:]
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertEqual([row["case"] for row in result["absent"]], ["dense_matmul"])
        self.assertIn("| (none) |", text)
        self.assertIn("Only one run recorded 1 row(s)", text)
        self.assertIn("dense_matmul/matmul/tenferro-cuda-eager (baseline only, status ok)", text)

    def test_a_row_only_the_candidate_recorded_is_named(self):
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate["rows"].append({"case": "new", "group": "matmul", "backend": "pytorch-cuda",
                                  "median": 1.0, "status": "ok", "verified": True})
        result = compare(candidate, baseline, 1.2)
        self.assertEqual([row["case"] for row in result["absent"]], ["new"])
        self.assertIn("new/matmul/pytorch-cuda (candidate only, status ok)", render(result))

    def test_a_measurement_only_one_run_recorded_exits_nonzero(self):
        # Naming the identity is not enough: the table no longer covers the run,
        # so a missing measurement must not read as a successful comparison.
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate['rows'] = candidate['rows'][1:]
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertIn('dense_matmul/matmul/tenferro-cuda-eager (baseline only, status ok)', text)
        self.assertEqual(status(result), 1)

    def test_a_skipped_row_the_other_run_never_recorded_keeps_the_status(self):
        # Neither week measured this identity, so nothing was lost: it is named,
        # but a run whose only gaps are recorded skips is still a comparison.
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        baseline['rows'][0] = {'case': 'dense_matmul', 'group': 'matmul',
                               'backend': 'tenferro-cuda-eager', 'status': 'skipped',
                               'reason': 'no kernel', 'median': None, 'iqr': None}
        candidate['rows'] = candidate['rows'][1:]
        result = compare(candidate, baseline, 1.2)
        self.assertIn('tenferro-cuda-eager (baseline only, status skipped)', render(result))
        self.assertEqual(status(result), 0)

    def test_an_invalid_candidate_without_a_baseline_is_still_named(self):
        # render() returned the missing-baseline message before reading the
        # invalid findings, so an invalid row vanished when the week had none.
        candidate = record(WEEK_45_LATE, COMMIT_B)
        candidate["rows"][0]["median"] = -1.0
        result = compare(candidate, None, 1.2)
        text = render(result)
        self.assertNotIn("| case |", text)
        self.assertIn("candidate dense_matmul/matmul/tenferro-cuda-eager: median is not a finite "
                      "non-negative number", text)
        self.assertIn("no baseline either", text)
        self.assertEqual(status(result), 1)


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

    def commit(self, message, date=None):
        self.git('add', '-A')
        environment = {**os.environ}
        if date:
            environment['GIT_AUTHOR_DATE'] = environment['GIT_COMMITTER_DATE'] = date
        subprocess.run(['git', '-C', str(self.repo), 'commit', '-q', '-m', message], check=True,
                       capture_output=True, env=environment)

    def sidecar(self, document):
        return f"result/{document['target_profile']}/{document['suite_id']}.summary.json"

    def receipt(self, relative, payload):
        return self.write(relative.replace('.summary.json', '.provenance.json'),
                          json.dumps({'files': {relative: hashlib.sha256(payload).hexdigest()}}).encode())

    def accept(self, document, date=None):
        """What the ingestion driver commits: the sidecar plus the receipt for its bytes."""
        payload = json.dumps(document).encode()
        relative = self.write(self.sidecar(document), payload)
        self.receipt(relative, payload)
        self.commit('Accept benchmark run 1 attempt 1', date=date)

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
        self.assertIn('| dense_matmul | matmul | tenferro-cuda-eager | 1 | 1.5 | 1.50x |', process.stdout)

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
        # A suite the driver never ingested, in a repository that does record
        # receipts: the run's report is published with the sidecar, so a
        # sidecar-only revision is an edit, not a run.
        ingested = record(WEEK_43, COMMIT_A)
        ingested['suite_id'] = 'gpu/ingested'
        self.accept(ingested)
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

    def test_an_unrelated_merged_receipt_does_not_move_the_history_boundary(self):
        # The boundary is whether *this* revision's own history carries a receipt.
        # A receipt committed on another branch and merged later reclassified an
        # already published run when the boundary was a single commit.
        self.write('README.md', b'bench\n')
        self.commit('Initial commit', date='2026-10-20T00:00:00+00:00')
        self.git('checkout', '-q', '-b', 'other')
        other = record(WEEK_43, COMMIT_A)
        other['suite_id'] = 'gpu/other'
        self.accept(other, date='2026-10-27T00:00:00+00:00')
        self.git('checkout', '-q', 'main')
        published = record(WEEK_44, COMMIT_B)
        self.write(self.sidecar(published), json.dumps(published).encode())
        self.commit('Write the report summary', date='2026-11-01T00:00:00+00:00')
        self.accept(record(WEEK_45_LATE, COMMIT_B, scale=1.5), date='2026-11-08T00:00:00+00:00')
        self.git('merge', '--no-ff', 'other', '-m', 'merge the other branch')
        records = accepted_runs(self.repo, 'nvidia-gpu', 'gpu/dense')
        self.assertEqual([item['timestamp'] for item in records], [WEEK_45_LATE, WEEK_44])

    def test_unverified_rows_are_named_instead_of_compared(self):
        # result_ingestion refuses a measurement whose `verified` is not true, so a
        # record that reached the history through the receipt-free path can carry
        # rows the driver rejected: they are not a 10x regression either.
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate['rows'][0].update(median=10.0, verified=False, verification='measure_only',
                                    reason='measured without a check')
        result = compare(candidate, baseline, 1.2)
        text = render(result)
        self.assertEqual(result['rows'], [])
        self.assertNotIn('10.00x', text)
        self.assertIn('Skipped 1 unverified row(s)', text)
        self.assertIn('dense_matmul/matmul/tenferro-cuda-eager (candidate unverified)', text)

    def test_a_verified_row_is_compared_alongside_a_skipped_one(self):
        baseline, candidate = record(WEEK_44, COMMIT_A), record(WEEK_45_LATE, COMMIT_B)
        candidate['rows'][0].update(median=10.0, verified=False)
        baseline['rows'][1]['median'] = 2.0
        candidate['rows'][1]['median'] = 10.0
        result = compare(candidate, baseline, 1.2)
        self.assertEqual([(row['backend'], row['ratio']) for row in result['rows']],
                         [('pytorch-cuda', 5.0)])
        self.assertIn('Skipped 1 unverified row(s)', render(result))

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
