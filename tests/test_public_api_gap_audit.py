from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_public_api_gaps", ROOT / "scripts/audit_public_api_gaps.py"
)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


class PublicApiGapAuditTest(unittest.TestCase):
    def test_every_public_session_method_has_benchmark_evidence(self) -> None:
        gaps = [api.name for api, hits in AUDIT.rows() if not hits]
        self.assertEqual(gaps, [])


if __name__ == "__main__":
    unittest.main()
