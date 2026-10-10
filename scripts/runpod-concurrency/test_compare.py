import copy,json,unittest
from pathlib import Path
from compare import evaluate
P=json.loads(Path('result/nvidia-gpu/ci/runpod-concurrency-protocol.json').read_text())
def rows():
 return [dict(arm=a,paid_seconds=350 if a=='A' else 240,driver='570.211.01',
              stages={'cuda_tests':{'seconds':260 if a=='A' else 150}},gpu_memory_peak_mib=2048) for a in P['order']]
class CompareTests(unittest.TestCase):
 def test_improvement(self):
  r=evaluate(rows(),P);self.assertEqual(r['verdict'],'PASS');self.assertGreater(r['paired_log_t_95_reduction_interval'][0],.1)
 def test_no_improvement(self):
  r=rows()
  for x in r:x['paid_seconds']=350
  self.assertEqual(evaluate(r,P)['verdict'],'FAIL')
 def test_missing_sample(self):
  with self.assertRaises(ValueError):evaluate(rows()[:-1],P)
 def test_aa_noise(self):
  r=rows();r[0]['paid_seconds']=500;self.assertEqual(evaluate(r,P)['verdict'],'INCONCLUSIVE')
 def test_imbalance(self):
  r=rows()
  for x in r:
   if x['arm']=='B':x['driver']='other'
  self.assertEqual(evaluate(r,P)['verdict'],'INCONCLUSIVE')
 def test_memory(self):
  r=rows();r[-1]['gpu_memory_peak_mib']=14000;self.assertEqual(evaluate(r,P)['verdict'],'FAIL')
 def test_cold_candidate_is_retained(self):
  r=rows();r[-1]['paid_seconds']=650;self.assertEqual(evaluate(r,P)['verdict'],'FAIL')
unittest.main()
