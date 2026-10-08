"""Failures and stale references must never become performance evidence."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import cpu_followup_cases as adapter
import generate_perf_issue_cases as generator

class FollowupTest(unittest.TestCase):
    def test_mismatching_reference_prevents_collection(self):
        sig=dict(shape=[1],count=1,sum=[1.,0.],sum_abs=1.,sum_sq=1.,probes=[[0,1.,0.]])
        bad=dict(sig,sum=[9.,0.],probes=[[0,9.,0.]])
        case=next(c for c in generator.generate_cases() if c['kind']=='cpu_followup_mwe')
        with patch.object(adapter,'execute',side_effect=[dict(status='ok',outputs=[sig]),dict(status='ok',outputs=[bad])]) as execute:
            result=adapter.run(case,1,dict(min_runtime_ms=10,runs=15))
        self.assertEqual(execute.call_count,2)
        self.assertEqual(result['correctness_status'],'failed')
        self.assertEqual(result['samples'],[])

if __name__=='__main__':unittest.main()
