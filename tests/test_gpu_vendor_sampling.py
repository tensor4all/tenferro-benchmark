import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import benchmark_gpu_python as vendors
import benchmark_gpu_public_api as sampler


class VendorSamplingTest(unittest.TestCase):
    def test_verify_only_never_enters_sampler_and_unwraps_native_output(self):
        tensor=types.SimpleNamespace(device='cuda:0',dtype='f64',numel=lambda:16,element_size=lambda:8)
        native_output=types.SimpleNamespace(tensor=tensor)
        torch=types.SimpleNamespace(float64='f64',set_num_threads=lambda n:None)
        with patch.dict(sys.modules,torch=torch),patch.dict('os.environ',GPU_BENCH_VERIFY_ONLY='1'),patch.object(sampler,'measure') as measure:
            sampling, verification=vendors._sample_vendor(lambda:native_output,lambda:None,
                lambda value:('passed',0.,0.) if value is tensor else ('failed',1.,1.),'cuda:0',0,3)
            measure.assert_not_called()
        self.assertIsNone(sampling)
        self.assertEqual(verification,('passed',0.,0.))

    def test_raw_samples_and_normalization_survive_record_creation(self):
        sampling=dict(operations_per_sample=2,batch_ms=[2.,4.,6.])
        first,median,_,_,_=vendors._vendor_stats(sampling)
        self.assertEqual((first,median),(1.,2.))
        record=dict(status='ok',backend='cutlass',execution={})
        result=vendors._vendor_record(record,sampling,'test vendor')
        notes=json.loads(result['execution']['notes'])
        self.assertEqual(notes['sampling'],sampling)
        self.assertIn('precision',notes)
        result=vendors._vendor_record(dict(status='ok',backend='cutlass',execution={}),None,'test vendor')
        self.assertEqual(result['status'],'verification_only')

    def test_provider_annotation_preserves_raw_samples_and_setup(self):
        original=dict(sampling=dict(operations_per_sample=1,batch_ms=[1.,2.,3.]),setup="untimed",precision="f64")
        record=dict(execution=dict(notes=json.dumps(original)))
        result=vendors._annotate_vendor(record,"preferred-provider","fallback allowed")
        notes=json.loads(result['execution']['notes'])
        self.assertEqual(notes['sampling'],original['sampling'])
        self.assertEqual(notes['setup'],'untimed')
        self.assertEqual(notes['vendor_path'],'fallback allowed')

    def test_oversized_single_output_has_an_explicit_effective_bound(self):
        clock=iter(range(0,1000000,100))
        with patch.object(sampler.time,'perf_counter_ns',side_effect=lambda:next(clock)):
            result=sampler.measure(lambda:object(),lambda:None,lambda values:None,
                                   sampler.RETENTION_BUDGET*2,0,3)
        self.assertEqual(result['operations_per_sample'],1)
        self.assertTrue(result['single_output_exceeds_configured_budget'])
        self.assertEqual(result['effective_retention_bound_bytes'],sampler.RETENTION_BUDGET*2)
        self.assertEqual(len(result['batch_ms']),3)


if __name__=='__main__':unittest.main()
