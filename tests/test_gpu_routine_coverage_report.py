import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from format_gpu_routine_coverage import render


class CoverageReportTest(unittest.TestCase):
    def test_verification_and_omissions_are_explicit(self):
        status = dict(manifest_version=1, coverage="quick", effort="scan", complete=False)
        for name in ("expected", "selected", "executed", "unsupported", "failed", "missing",
                     "verification_only", "raw_samples_unavailable", "noisy", "unselected",
                     "duplicates", "unexpected"):
            status[name] = []
        status.update(expected=["s:a:b"], selected=["s:a:b"],
                      verification_only=["s:a:b"], unselected=["s:large:b"])
        status["counts"] = {key: len(value) for key, value in status.items() if isinstance(value, list)}
        report = render(status)
        self.assertIn("Measured coverage complete: no.", report)
        self.assertIn("| executed | 0 |", report)
        self.assertIn("## verification_only\n\n- `s:a:b`", report)
        self.assertIn("## unselected\n\n- `s:large:b`", report)


if __name__ == "__main__":
    unittest.main()
