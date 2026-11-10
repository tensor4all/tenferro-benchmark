import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from scripts.ci import weekly_gate
from scripts.ci.weekly_gate import (already_collected, budget_report, count_purposes,
                                    normalize_purpose, purpose_of, refusal, week_start)


def at(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def this_week(hours):
    """A timestamp inside the current weekly window, `hours` after it opened."""
    start = week_start(datetime.now(timezone.utc))
    return (start + timedelta(hours=hours)).strftime('%Y-%m-%dT%H:%M:%SZ')


def read(path):
    if not os.path.exists(path):
        return ''
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def gpu_run(run_id, created, purpose='', event='workflow_dispatch', conclusion='success',
            branch='main', attempt=1):
    title = 'RunPod GPU benchmark'
    if purpose:
        title += f' [purpose={purpose} target=SpM-lab/tenferro-rs#1]'
    return {'databaseId': run_id, 'attempt': attempt, 'createdAt': created, 'event': event,
            'conclusion': conclusion, 'displayTitle': title, 'headBranch': branch}


class WeeklyGateTests(unittest.TestCase):
    def test_week_starts_on_saturday_utc(self):
        self.assertEqual(week_start(at('2026-10-17T20:00:00')), at('2026-10-17T00:00:00'))
        self.assertEqual(week_start(at('2026-10-18T08:00:00')), at('2026-10-17T00:00:00'))
        self.assertEqual(week_start(at('2026-10-16T23:59:00')), at('2026-10-10T00:00:00'))

    def test_success_this_week_skips_later_slots(self):
        runs = [{'databaseId': 1, 'createdAt': '2026-10-17T08:00:05Z', 'event': 'schedule',
                 'conclusion': 'success', 'headBranch': 'main',
                 'displayTitle': 'RunPod GPU benchmark [purpose=weekly target=-]'}]
        self.assertTrue(already_collected(runs, at('2026-10-18T08:00:00'), '2'))

    def test_last_week_or_own_run_does_not_count(self):
        runs = [{'databaseId': 1, 'createdAt': '2026-10-11T08:00:05Z', 'event': 'schedule',
                 'conclusion': 'success', 'headBranch': 'main',
                 'displayTitle': 'RunPod GPU benchmark'},
                {'databaseId': 2, 'createdAt': '2026-10-17T14:00:05Z', 'event': 'schedule',
                 'conclusion': 'success', 'headBranch': 'main',
                 'displayTitle': 'RunPod GPU benchmark'}]
        self.assertFalse(already_collected(runs, at('2026-10-17T14:00:30'), '2'))

    def test_a_slot_that_collected_still_counts_after_its_re_run(self):
        # A re-run of a collected slot is refused, and that refusal leaves the run with
        # conclusion=failure while its first attempt keeps the week's results.
        runs = [gpu_run(1, '2026-10-17T08:00:05Z', event='schedule', conclusion='failure',
                        attempt=2)]
        self.assertFalse(already_collected(runs, at('2026-10-18T08:00:00'), '2'))
        runs[0]['prior_success'] = True
        self.assertTrue(already_collected(runs, at('2026-10-18T08:00:00'), '2'))

    def test_a_week_of_verifications_is_not_a_collected_week(self):
        # A verify run checks a fix; reading it as a satisfied week would hide a
        # missed collection instead of reporting it.
        runs = [gpu_run(1, '2026-10-17T08:00:05Z', 'verify', event='schedule'),
                gpu_run(2, '2026-10-18T09:00:05Z', 'verify')]
        self.assertFalse(already_collected(runs, at('2026-10-18T10:00:00'), '3'))


class PurposeTests(unittest.TestCase):
    def test_a_run_without_a_purpose_counts_as_weekly(self):
        self.assertEqual(normalize_purpose(''), 'weekly')
        self.assertEqual(normalize_purpose(None), 'weekly')
        self.assertEqual(normalize_purpose('verfy'), 'weekly')
        self.assertEqual(purpose_of({'displayTitle': 'RunPod GPU benchmark'}), 'weekly')
        self.assertEqual(purpose_of({'displayTitle': 'RunPod GPU benchmark [purpose=backfill target=-]'}),
                         'backfill')
        counts = count_purposes([gpu_run(1, '2026-10-17T08:00:05Z'),
                                 gpu_run(2, '2026-10-17T09:00:05Z', 'backfill')],
                                at('2026-10-18T10:00:00'), '3')
        self.assertEqual(counts, {'weekly': 1, 'verify': 0, 'backfill': 1})

    def test_a_scheduled_slot_does_not_spend_the_dispatch_budget(self):
        # Four slots per weekend, and the ones that skip never rent a GPU: counting
        # them would spend the weekly cap on a normal week.
        runs = [gpu_run(1, '2026-10-17T08:00:05Z', event='schedule'),
                gpu_run(2, '2026-10-17T14:00:05Z', event='schedule'),
                gpu_run(3, '2026-10-17T20:00:05Z', event='schedule'),
                gpu_run(4, '2026-10-18T08:00:05Z', event='schedule')]
        counts = count_purposes(runs, at('2026-10-18T10:00:00'), '5')
        self.assertEqual(counts, {'weekly': 0, 'verify': 0, 'backfill': 0})

    def test_the_weekly_and_backfill_cap_refuses_the_fourth_run(self):
        counts = {'weekly': 2, 'backfill': 1, 'verify': 0}
        self.assertIsNone(refusal('weekly', '', {'weekly': 2, 'backfill': 0, 'verify': 0}))
        refused = refusal('backfill', '', counts)
        self.assertIn('weekly budget spent this week (weekly 3/3, verify 0/6)', refused)

    def test_the_verify_cap_allows_six_and_refuses_the_seventh(self):
        target = 'tensor4all/tenferro-rs#419'
        self.assertIsNone(refusal('verify', target, {'weekly': 0, 'backfill': 0, 'verify': 5}))
        refused = refusal('verify', target, {'weekly': 0, 'backfill': 0, 'verify': 6})
        self.assertIn('verify budget spent this week (weekly 0/3, verify 6/6)', refused)

    def test_a_verification_must_name_what_it_verifies(self):
        counts = {'weekly': 0, 'backfill': 0, 'verify': 2}
        self.assertIn('needs target=<repo>#<n>', refusal('verify', '', counts))
        self.assertIn('needs target=<repo>#<n>', refusal('verify', '#419', counts))
        # The other purposes do not need a target.
        self.assertIsNone(refusal('weekly', '', counts))

    def test_the_weekly_report_prints_both_counters(self):
        counts = {'weekly': 1, 'backfill': 0, 'verify': 4}
        self.assertEqual(budget_report(counts), 'weekly 1/3, verify 4/6')


class GateMainTests(unittest.TestCase):
    def run_gate(self, runs, purpose='', target='', event='workflow_dispatch', run_attempt='1',
                 branch='main', attempts=()):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {'output': os.path.join(tmp, 'output'), 'summary': os.path.join(tmp, 'summary')}
            environment = {'GITHUB_OUTPUT': paths['output'], 'GITHUB_STEP_SUMMARY': paths['summary']}
            argv = ['weekly_gate', '--repository', 'tensor4all/tenferro-benchmark',
                    '--event', event, '--run-id', '99', '--run-attempt', run_attempt,
                    '--branch', branch, '--purpose', purpose, '--target', target]
            answers = [json.dumps(runs)] + list(attempts)
            buffer = io.StringIO()
            with patch.object(sys, 'argv', argv), patch.dict(os.environ, environment), \
                    patch.object(weekly_gate.subprocess, 'check_output', side_effect=answers), \
                    redirect_stdout(buffer):
                code = 0
                try:
                    weekly_gate.main()
                except SystemExit as stop:
                    code = stop.code
            written = {name: read(path) for name, path in paths.items()}
        return code, buffer.getvalue(), written['output'], written['summary']

    def test_a_mixed_week_reports_both_counters_and_proceeds(self):
        runs = [gpu_run(1, this_week(1), 'weekly')] + \
               [gpu_run(10 + index, this_week(2 + index), 'verify') for index in range(4)]
        code, stdout, output, summary = self.run_gate(runs, purpose='weekly')
        self.assertEqual(code, 0)
        self.assertIn('Dispatch budget this week: weekly 1/3, verify 4/6', stdout)
        self.assertIn('proceed=true', output)
        self.assertEqual(summary.count('weekly 1/3, verify 4/6'), 1)
    def test_the_fourth_weekly_dispatch_is_refused_with_the_reason(self):
        runs = [gpu_run(1, this_week(1), 'weekly'),
                gpu_run(2, this_week(2), 'weekly'),
                gpu_run(3, this_week(3), 'backfill')]
        code, stdout, output, summary = self.run_gate(runs, purpose='weekly')
        self.assertEqual(code, 1)
        self.assertIn('weekly budget spent this week (weekly 3/3, verify 0/6)', stdout)
        self.assertIn('refused', summary)
        self.assertNotIn('proceed=', output)

    def test_the_seventh_verification_is_refused_with_the_reason(self):
        runs = [gpu_run(10 + index, this_week(1 + index), 'verify') for index in range(6)]
        code, stdout, _, _ = self.run_gate(runs, purpose='verify', target='tensor4all/tenferro-rs#419')
        self.assertEqual(code, 1)
        self.assertIn('verify budget spent this week (weekly 0/3, verify 6/6)', stdout)

    def test_a_verification_without_a_target_is_refused(self):
        code, stdout, _, _ = self.run_gate([], purpose='verify')
        self.assertEqual(code, 1)
        self.assertIn('purpose=verify needs target=<repo>#<n>', stdout)

    def test_the_sixth_verification_proceeds(self):
        runs = [gpu_run(10 + index, this_week(1 + index), 'verify') for index in range(5)]
        code, stdout, output, summary = self.run_gate(runs, purpose='verify',
                                                     target='tensor4all/tenferro-rs#419')
        self.assertEqual(code, 0)
        self.assertIn('proceed=true', output)
        self.assertIn('purpose=verify target=tensor4all/tenferro-rs#419: weekly 0/3, verify 5/6', summary)

    def test_a_queued_successor_does_not_spend_the_budget_of_this_dispatch(self):
        # A run that is still queued or running may not have rented yet; the record
        # arrives the moment it is queued.
        runs = [gpu_run(1, this_week(1), 'weekly'), gpu_run(2, this_week(2), 'weekly'),
                gpu_run(500, this_week(3), 'weekly', conclusion=None)]
        code, stdout, output, _ = self.run_gate(runs, purpose='weekly')
        self.assertEqual(code, 0)
        self.assertIn('Dispatch budget this week: weekly 2/3, verify 0/6', stdout)
        self.assertIn('proceed=true', output)

    def test_a_finished_successor_spends_the_budget(self):
        # The concurrency group does not order runs by id: 100-102 can finish before 99
        # starts, and their A100 hours are spent whether or not their ids are higher.
        runs = [gpu_run(100, this_week(1), 'weekly'), gpu_run(101, this_week(2), 'weekly'),
                gpu_run(102, this_week(3), 'backfill')]
        code, stdout, output, _ = self.run_gate(runs, purpose='weekly')
        self.assertEqual(code, 1)
        self.assertIn('weekly budget spent this week (weekly 3/3, verify 0/6)', stdout)
        self.assertNotIn('proceed=', output)

    def test_mixed_successors_count_only_when_they_finished(self):
        runs = [gpu_run(100, this_week(1), 'weekly', conclusion=None),
                gpu_run(101, this_week(2), 'weekly'),
                gpu_run(102, this_week(3), 'verify')]
        code, stdout, _, _ = self.run_gate(runs, purpose='weekly')
        self.assertEqual(code, 0)
        self.assertIn('Dispatch budget this week: weekly 1/3, verify 1/6', stdout)

    def test_a_requeued_successor_still_spends_the_budget(self):
        # Re-running a run clears its conclusion while the new attempt runs, but its
        # finished attempt already spent; a first attempt that is still queued has not.
        runs = [gpu_run(100, this_week(1), 'weekly', conclusion=None, attempt=2),
                gpu_run(101, this_week(2), 'weekly', conclusion=None)]
        code, stdout, output, _ = self.run_gate(runs, purpose='weekly')
        self.assertEqual(code, 0)
        self.assertIn('Dispatch budget this week: weekly 1/3, verify 0/6', stdout)
        self.assertIn('proceed=true', output)

    def test_a_rerun_of_a_dispatch_is_refused(self):
        code, stdout, _, _ = self.run_gate([], purpose='weekly', run_attempt='2')
        self.assertEqual(code, 1)
        self.assertIn('a re-run repeats a run that is already recorded', stdout)

    def test_a_rerun_never_rents(self):
        # What the repeated attempt actually did cannot be told: a failure can follow a
        # successful collection, and a failed provisioning can leave a pod billing. A
        # scheduled slot retries at its next slot; an immediate retry is a counted
        # dispatch.
        for event in ('schedule', 'workflow_dispatch'):
            for run_attempt in ('2', '3'):
                with self.subTest(event=event, run_attempt=run_attempt):
                    code, stdout, output, _ = self.run_gate([], event=event, run_attempt=run_attempt)
                    self.assertEqual(code, 1)
                    self.assertIn('a retry that may rent has to be counted', stdout)
                    self.assertNotIn('proceed=', output)

    def test_a_rerun_counts_the_runs_that_came_after_it(self):
        # A re-run executes after the runs it has to be counted against, so a later
        # dispatch is not a queued successor for it.
        runs = [gpu_run(500, this_week(3), 'weekly'), gpu_run(600, this_week(4), 'weekly')]
        code, stdout, _, _ = self.run_gate(runs, event='schedule', run_attempt='2')
        self.assertIn('Dispatch budget this week: weekly 2/3, verify 0/6', stdout)

    def test_a_truncated_history_fails_closed(self):
        runs = [gpu_run(1 + index, this_week(1), 'verify') for index in range(weekly_gate.WINDOW_LIMIT)]
        code, stdout, _, _ = self.run_gate(runs, purpose='verify', target='tensor4all/tenferro-rs#419')
        self.assertEqual(code, 1)
        self.assertIn('fail closed rather than rent against an unknown budget', stdout)

    def test_the_schedule_still_skips_a_collected_week(self):
        runs = [gpu_run(2, this_week(1), event='schedule')]
        code, stdout, output, _ = self.run_gate(runs, event='schedule')
        self.assertEqual(code, 0)
        self.assertIn('This week already has successful GPU results; skipping.', stdout)
        self.assertIn('proceed=false', output)

    def test_a_development_branch_record_does_not_spend_mains_budget(self):
        # A dispatch from a feature branch records the same workflow in the same
        # weekly window. main's scheduled slot must not read that spending as its
        # own and refuse itself; the counters it prints must be main's too.
        runs = [gpu_run(1 + index, this_week(1 + index), 'weekly', branch='feat/perf-probe')
                for index in range(3)] + [gpu_run(10, this_week(6), 'weekly')]
        code, stdout, output, summary = self.run_gate(runs, purpose='weekly')
        self.assertEqual(code, 0)
        self.assertIn('Dispatch budget this week: weekly 1/3, verify 0/6', stdout)
        self.assertIn('weekly 1/3, verify 0/6', summary)
        self.assertIn('proceed=true', output)

    def test_a_development_branch_success_is_not_a_collected_week(self):
        runs = [gpu_run(1, this_week(1), event='schedule', branch='feat/perf-probe')]
        code, stdout, output, _ = self.run_gate(runs, event='schedule')
        self.assertEqual(code, 0)
        self.assertIn('No successful GPU run this week yet; collecting.', stdout)
        self.assertIn('proceed=true', output)

    def test_a_feature_branch_counts_only_its_own_dispatches(self):
        # Per-branch scoping is not an exemption: a fix verified from a branch still
        # spends that branch's cap, while main's records do not enter its counters.
        runs = [gpu_run(1, this_week(1), 'weekly'),
                gpu_run(2, this_week(2), 'verify', branch='feat/perf-probe'),
                gpu_run(3, this_week(3), 'verify', branch='feat/perf-probe')]
        code, stdout, _, _ = self.run_gate(runs, purpose='verify', target='SpM-lab/tenferro-rs#1',
                                           branch='feat/perf-probe')
        self.assertEqual(code, 0)
        self.assertIn('Dispatch budget this week: weekly 0/3, verify 2/6', stdout)

    def test_the_history_query_is_bounded_to_the_branch_and_carries_its_fields(self):
        # The window is bounded to this branch on the server: another branch's runs
        # in the 100-run limit would fail main closed while it spent nothing. The
        # branch and the attempt count are what the counters are read from.
        call = MagicMock(return_value='[]')
        argv = ['weekly_gate', '--repository', 'tensor4all/tenferro-benchmark',
                '--event', 'workflow_dispatch', '--run-id', '99', '--branch', 'feat/perf-probe']
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, {}, clear=True), \
                patch.object(weekly_gate.subprocess, 'check_output', call), \
                redirect_stdout(io.StringIO()):
            weekly_gate.main()
        query = call.call_args.args[0]
        self.assertEqual(query[query.index('--branch') + 1], 'feat/perf-probe')
        self.assertIn('headBranch', query[query.index('--json') + 1])
        self.assertIn('attempt', query[query.index('--json') + 1])

    def test_a_collected_slot_keeps_counting_after_its_refused_re_run(self):
        # 1. a slot collects on its first attempt; 2. "Re-run all jobs" is refused, which
        # leaves that run failed; 3. the next slot must still see the collected week.
        runs = [gpu_run(1, this_week(1), event='schedule', conclusion='failure', attempt=2)]
        code, stdout, output, _ = self.run_gate(runs, event='schedule', attempts=['success'])
        self.assertEqual(code, 0)
        self.assertIn('This week already has successful GPU results; skipping.', stdout)
        self.assertIn('proceed=false', output)

    def test_a_re_run_of_a_failed_slot_does_not_make_the_week_collected(self):
        # The negative side of the same fold: nothing succeeded, so the week is still
        # open and this slot collects.
        runs = [gpu_run(1, this_week(1), event='schedule', conclusion='failure', attempt=2)]
        code, stdout, output, _ = self.run_gate(runs, event='schedule', attempts=['failure'])
        self.assertEqual(code, 0)
        self.assertIn('No successful GPU run this week yet; collecting.', stdout)
        self.assertIn('proceed=true', output)

    def test_a_decided_week_does_not_query_the_remaining_attempts(self):
        # A success in an earlier attempt answers the question; an attempt that cannot be
        # read afterwards must not fail the gate instead of skipping the week.
        runs = [gpu_run(1, this_week(1), event='schedule', conclusion='failure', attempt=3)]
        code, stdout, _, _ = self.run_gate(runs, event='schedule',
                                           attempts=['success', RuntimeError('must not query')])
        self.assertEqual(code, 0)
        self.assertIn('This week already has successful GPU results; skipping.', stdout)

    def test_the_first_attempt_of_a_re_run_is_read(self):
        runs = [gpu_run(1, this_week(1), event='schedule', conclusion='failure', attempt=2)]
        call = MagicMock(side_effect=[json.dumps(runs), 'success'])
        argv = ['weekly_gate', '--repository', 'tensor4all/tenferro-benchmark',
                '--event', 'schedule', '--run-id', '99']
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, {}, clear=True), \
                patch.object(weekly_gate.subprocess, 'check_output', call), \
                redirect_stdout(io.StringIO()):
            weekly_gate.main()
        query = call.call_args_list[1].args[0]
        self.assertEqual(query[2], 'repos/tensor4all/tenferro-benchmark/actions/runs/1/attempts/1')

    def test_a_later_slot_that_already_collected_is_not_re_rented(self):
        # The concurrency group serializes the weekend slots but does not order them
        # by run id, so a later slot can finish the collection first. Counting only
        # predecessors would rent a second pod for a week that is already collected.
        runs = [gpu_run(500, this_week(2), event='schedule')]
        code, stdout, output, _ = self.run_gate(runs, event='schedule')
        self.assertEqual(code, 0)
        self.assertIn('This week already has successful GPU results; skipping.', stdout)
        self.assertIn('proceed=false', output)

    def test_a_spent_budget_refuses_the_scheduled_slot_too(self):
        runs = [gpu_run(1, this_week(1), 'weekly'),
                gpu_run(2, this_week(2), 'weekly'),
                gpu_run(3, this_week(3), 'weekly')]
        code, stdout, output, _ = self.run_gate(runs, event='schedule')
        self.assertEqual(code, 1)
        self.assertIn('weekly budget spent this week (weekly 3/3, verify 0/6)', stdout)
        self.assertNotIn('proceed=', output)


if __name__ == '__main__':
    unittest.main()
