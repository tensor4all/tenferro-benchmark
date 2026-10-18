import unittest
from datetime import datetime, timezone

from scripts.ci.weekly_gate import already_collected, week_start


def at(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


class WeeklyGateTests(unittest.TestCase):
    def test_week_starts_on_saturday_utc(self):
        self.assertEqual(week_start(at('2026-10-17T20:00:00')), at('2026-10-17T00:00:00'))
        self.assertEqual(week_start(at('2026-10-18T08:00:00')), at('2026-10-17T00:00:00'))
        self.assertEqual(week_start(at('2026-10-16T23:59:00')), at('2026-10-10T00:00:00'))

    def test_success_this_week_skips_later_slots(self):
        runs = [{'databaseId': 1, 'createdAt': '2026-10-17T08:00:05Z'}]
        self.assertTrue(already_collected(runs, at('2026-10-18T08:00:00'), '2'))

    def test_last_week_or_own_run_does_not_count(self):
        runs = [{'databaseId': 1, 'createdAt': '2026-10-11T08:00:05Z'},
                {'databaseId': 2, 'createdAt': '2026-10-17T14:00:05Z'}]
        self.assertFalse(already_collected(runs, at('2026-10-17T14:00:30'), '2'))


if __name__ == '__main__':
    unittest.main()
