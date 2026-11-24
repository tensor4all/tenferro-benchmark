"""A page that mixes tenferro-rs revisions says so (#515)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_dashboard as dashboard  # noqa: E402

NEW = "ce9125d0cff4ee40841120759b5df45252c27f22"
OLD = "9a850b06155405acc99ecd0648829e9e6d3866ba"


def doc(suite, commit, stamp):
    return {"suite_id": suite, "tenferro_commit": commit, "timestamp": stamp}


class RevisionLabelTest(unittest.TestCase):
    def test_one_revision_keeps_the_single_commit_header(self):
        docs = [doc("cpu/einsum", NEW, "2026-11-24T04:00:00Z"), doc("cpu/fft", NEW, "2026-11-24T05:00:00Z")]
        self.assertEqual(dashboard.revision_label(docs), "<code>ce9125d0cff4</code>")
        self.assertEqual(dashboard.mixed_revisions_notice(docs), "")

    def test_a_mixed_page_names_every_revision_newest_first(self):
        docs = [
            doc("cpu/public_api", OLD, "2026-11-12T20:13:05Z"),
            doc("cpu/einsum", NEW, "2026-11-24T04:52:02Z"),
            doc("cpu/fft", NEW, "2026-11-24T05:30:27Z"),
        ]
        self.assertEqual(
            dashboard.revision_label(docs),
            "<code>ce9125d0cff4</code> (2 suites) · <code>9a850b061554</code> (1 suite)",
        )
        notice = dashboard.mixed_revisions_notice(docs)
        self.assertIn("Mixed tenferro-rs revisions", notice)
        self.assertIn("<code>9a850b061554</code>: cpu/public_api", notice)
        self.assertIn("not a like-for-like comparison", notice)

    def test_no_commit_no_label(self):
        self.assertIsNone(dashboard.revision_label([{"suite_id": "cpu/einsum"}, None]))


class RevisionChangeNoticeTest(unittest.TestCase):
    def view(self, suite, commit):
        return dashboard.SuiteView("zen5-cpu", suite, Path("x.md"), doc(suite, commit, "2026-11-24T00:00:00Z"), [])

    def test_a_suite_whose_baseline_is_another_revision_is_named(self):
        views = [self.view("cpu/einsum", NEW), self.view("cpu/public_api", OLD)]
        previous = {("zen5-cpu", "cpu/einsum"): OLD, ("zen5-cpu", "cpu/public_api"): OLD}
        notice = dashboard.revision_change_notice(views, previous)
        self.assertIn("cpu/einsum: <code>9a850b061554</code> &rarr; <code>ce9125d0cff4</code>", notice)
        self.assertNotIn("cpu/public_api", notice)
        self.assertIn("not comparable with the previous collection", notice)

    def test_same_revision_or_no_baseline_says_nothing(self):
        views = [self.view("cpu/einsum", NEW)]
        self.assertEqual(dashboard.revision_change_notice(views, {("zen5-cpu", "cpu/einsum"): NEW}), "")
        self.assertEqual(dashboard.revision_change_notice(views, {}), "")
        self.assertEqual(dashboard.revision_change_notice(views, None), "")


if __name__ == "__main__":
    unittest.main()
