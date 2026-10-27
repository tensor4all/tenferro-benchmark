"""Small CPU execution of CUDA reference fixtures; GPU correctness is separate."""
import unittest
from scripts.gpu_perf_issue_references import fixture


class ReferenceFixtureTest(unittest.TestCase):
    def test_fixtures_and_oracles(self):
        import torch
        definitions = [
            ('prepared_decode', dict(in_=7, out=5, len=3), (3,5)),
            ('standard_batched_matmul', dict(n=3, batch=2), (2,3,3)),
            ('linalg_readback_latency', dict(n=4, op='solve'), (4,1)),
            ('linalg_readback_latency', dict(n=4, op='qr'), (4,4)),
        ]
        for kind, params, shape in definitions:
            with self.subTest(kind=kind, params=params):
                if 'in_' in params:
                    params['in'] = params.pop('in_')
                op, check, retained_bytes = fixture(dict(kind=kind, params=params), 'cpu')
                out = op()
                check(out)
                if isinstance(out, torch.Tensor):
                    self.assertEqual(tuple(out.shape), shape)
                    self.assertEqual(out.numel()*out.element_size(), retained_bytes)
                    with self.assertRaises(AssertionError):
                        check(out + 1)
                else:
                    self.assertEqual(tuple(out[0].shape), shape)
                    self.assertEqual(sum(t.numel()*t.element_size() for t in out), retained_bytes)
                    with self.assertRaises(AssertionError):
                        check((out[0]+1, out[1]))

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            fixture(dict(kind='missing', params={}), 'cpu')
