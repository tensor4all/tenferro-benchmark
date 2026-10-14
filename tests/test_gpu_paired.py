import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import benchmark_gpu_paired as gpu
import detect_regressions as detector


class PairedTest(unittest.TestCase):
    identity=dict(device_uuid='GPU-one', device_name='test GPU', driver_version='driver', runtime_version='runtime')

    def campaign(self, root, arms=('a','b'), values=(100,100)):
        order=[]
        for rnd in (1,2):
            for arm in (arms if rnd==1 else arms[::-1]):
                folder=root/f'r{rnd}_{arm}'; folder.mkdir(parents=True)
                metadata=dict(cuda=self.identity, tenferro_rs=dict(commit='library',features=['native','cuda'],dirty=False),
                    harness=dict(commit='harness',dirty=False),collection=dict(manifest_version=5))
                (folder/'run_t1.yaml').write_text(json.dumps(metadata))
                row=dict(case_id='selected',threads=1,status='passed',kind='shape_stream',params=dict(n=16),
                         sequence=[dict(key=0)],samples=[dict(elapsed_ns=values[arms.index(arm)],iterations=1)]*3)
                (folder/'samples_t1.jsonl').write_text(json.dumps(row)+'\n')
                # Global suite omissions remain explicit, separate from the declared confirmation subset.
                (folder/'case_status.json').write_text(json.dumps(dict(manifest_version=5,not_selected=['other@t1'])))
                order.append(dict(round=rnd,arm=arm,dir=folder.name))
        (root/'plan.json').write_text(json.dumps(dict(order=order,harness_commit='harness',repetitions=dict(runs=3))))
        return detector.load_paired(root)

    def test_subset_does_not_hide_missing_or_changed_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            paired=self.campaign(Path(tmp))
            validate=lambda:gpu.validate_campaign(paired,['selected'],self.identity,dict(a='library',b='library'))
            self.assertEqual(validate(),[])
            paired['arms']['b'][2]['rows']['selected@t1']['sequence']=[dict(key=1)]
            self.assertTrue(any(f['kind']=='CHANGED_BASELINE' for f in validate()))
            del paired['arms']['a'][1]
            self.assertTrue(any(f['kind']=='MISSING' for f in validate()))

    def test_hardware_and_nonfinite_samples_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            paired=self.campaign(Path(tmp))
            run=paired['arms']['a'][1]
            run['rows']['selected@t1']['per_op']=[float('nan')]
            meta=Path(run['dir'])/'run_t1.yaml'; content=json.loads(meta.read_text())
            content['cuda']['device_uuid']='GPU-other';meta.write_text(json.dumps(content))
            findings=gpu.validate_campaign(paired,['selected'],self.identity,dict(a='library',b='library'))
            self.assertEqual({f['kind'] for f in findings},{'NUMERICAL_FAILURE','CHANGED_BASELINE'})

    def test_balanced_known_regression_reaches_shared_detector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); aa=self.campaign(root/'aa')
            paired=self.campaign(root/'paired',('baseline','candidate'),(100,150))
            config=dict(library=dict(baseline_commit='library',candidate_commit='library'),harness_commit='harness',
                        cases=dict(manifest_version=5),repetitions=dict(rounds=2),
                        thresholds=dict(relative=.1,absolute_ns=10),noise=dict(max_cov=.1,max_aa_relative_spread=.05))
            findings, verdicts=detector.confirm(config,paired,aa)
            self.assertEqual(findings,[])
            self.assertEqual(verdicts[0]['verdict'],'REGRESSION')

    def test_invalid_gpu_declaration_is_rejected(self):
        config=dict(status='declared',threads=[1],build=dict(profile='release',features='native,cuda'),
                    cases=dict(suite_id='gpu/perf_issues',case_ids=['selected']),
                    repetitions=dict(rounds=2,runs=15,warmups=3),host=dict(target_profile='nvidia-gpu',cuda=self.identity),
                    thresholds=dict(relative=.1,absolute_ns=1000),noise=dict(max_cov=.1,max_aa_relative_spread=.05))
        gpu.validate_config(config)
        for field,value in [('rounds',3),('runs',2),('warmups',-1)]:
            broken=copy.deepcopy(config);broken['repetitions'][field]=value
            with self.assertRaises(ValueError):gpu.validate_config(broken)
        config['noise']['max_cov']=float('nan')
        with self.assertRaises(ValueError):gpu.validate_config(config)


if __name__=='__main__':unittest.main()
