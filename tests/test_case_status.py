from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import case_status as cs


class CompletionTests(unittest.TestCase):
    def status(self, rows):
        return cs.build_case_status(suite_id="cpu/test", manifest_version=1,
                                    coverage="quick", effort="scan",
                                    expected=["a"], selected=["a"], rows=rows,
                                    selection_filter=None)

    def test_only_successful_complete_coverage_is_complete(self):
        passed = {"case_key": "a", "status": "passed"}
        self.assertTrue(self.status([passed])["complete"])
        for rows in ([], [{"case_key": "a", "status": "unsupported"}],
                     [{"case_key": "a", "status": "failed"}],
                     [{"case_key": "a", "status": "skipped"}], [passed, passed],
                     [passed, {"case_key": "b", "status": "passed"}]):
            self.assertFalse(self.status(rows)["complete"])

    def test_legacy_accounting_completion_does_not_mean_coverage(self):
        old = self.status([{"case_key": "a", "status": "unsupported"}])
        old["complete"] = True
        self.assertFalse(cs.merge_case_status([old])["complete"])

    def test_skip_is_accounted_but_never_passed_or_failed(self):
        status = self.status([{'case_key':'a','status':'skipped'}])
        self.assertEqual(status['counts']['skipped'],1)
        for field in ('executed','failed','missing'):
            self.assertEqual(status[field],[])
        self.assertFalse(status['complete'])
        self.assertIn('| expected | selected | executed | unsupported | failed | skipped |',
                      '\n'.join(cs.markdown_section(status)))
        old = self.status([{'case_key':'a','status':'failed'}])
        old.pop('skipped'); old['counts'].pop('skipped')
        self.assertNotIn('skipped',cs.merge_case_status([old]))
        self.assertNotIn('skipped','\n'.join(cs.markdown_section(old)))
        merged = cs.merge_case_status([old,status])
        self.assertEqual(merged['counts']['skipped'],1)
        self.assertEqual(merged['counts']['failed'],1)
        self.assertFalse(merged['complete'])


if __name__ == "__main__":
    unittest.main()
