import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.ci.check_gpu_run import bundle_run, digest
from scripts.ci.result_ingestion import plan_updates, validate_bundle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from format_gpu_results import build_summary


class GPUIngestionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/'data/results/nvidia-gpu/gpu/dense/20261018_000000'
        self.run.mkdir(parents=True)
        self.metadata = {'target_profile': 'nvidia-gpu', 'suite_id': 'gpu/dense',
                         'timestamp': '2026-10-18T00:00:00+00:00',
                         'tenferro_rs': {'commit': 'b'*40},
                         'cuda': {'device_name': 'NVIDIA A100 80GB PCIe'}}
        self.row = {'suite_id': 'gpu/dense', 'problem_id': 'matmul', 'backend': 'tenferro-cuda-trace',
                    'op': 'einsum', 'status': 'ok', 'timing': {'median_ms': 1.0, 'iqr_ms': 0.1, 'timed_runs': 3},
                    'verification': {'status': 'passed'}}
        (self.root/'dense.yaml').write_text('suite_id: gpu/dense\nproblems: [{id: matmul}]\n')
        (self.run/'run.yaml').write_text(yaml.safe_dump(self.metadata))
        (self.run/'records.jsonl').write_text(json.dumps(self.row)+'\n')
        (self.run/'report.md').write_text('Measured report\n')
        (self.run/'report.summary.json').write_text(json.dumps(build_summary([self.row], self.metadata))+'\n')
        self.bundle = self.root/'bundle'
        bundle_run(self.root, '20261018_000000', [Path('dense.yaml')], ['tenferro-cuda-trace'], self.bundle)
        self.source = {'head_sha': 'a'*40, 'run_id': 123, 'run_attempt': 1, 'artifact_id': 456}

    def validate(self):
        return validate_bundle(self.bundle, 'nvidia-gpu', self.source)

    def rewrite_summary(self, change):
        paths = ['result/nvidia-gpu/gpu/dense.summary.json',
                 'data/results/nvidia-gpu/gpu/dense/20261018_000000/report.summary.json']
        publication = json.loads((self.bundle/'publication.json').read_text())
        for relative in paths:
            path = self.bundle/relative
            document = json.loads(path.read_text())
            change(document)
            path.write_text(json.dumps(document)+'\n')
            publication['suites'][0]['files'][relative] = digest(path)
        (self.bundle/'publication.json').write_text(json.dumps(publication))

    def test_real_gpu_contract_is_accepted_without_mutating_bundle(self):
        before = {str(path.relative_to(self.bundle)): path.read_bytes() for path in self.bundle.rglob('*') if path.is_file()}
        validated = self.validate()
        updates, decisions = plan_updates(self.root/'repository', validated)
        self.assertEqual(decisions, {'gpu/dense': 'accepted'})
        self.assertEqual(len(updates), 3)
        self.assertFalse(any(name.startswith('data/') for name in updates))
        self.assertEqual(before, {str(path.relative_to(self.bundle)): path.read_bytes() for path in self.bundle.rglob('*') if path.is_file()})

    def test_redigested_summary_with_forged_value_is_rejected(self):
        self.rewrite_summary(lambda doc: doc['rows'][0].update(median=999))
        with self.assertRaisesRegex(ValueError, 'differs from raw'):
            self.validate()

    def test_consistent_other_a100_model_is_accepted_and_non_a100_rejected(self):
        self.set_model('NVIDIA A100-SXM4-40GB')
        self.validate()
        self.set_model('NVIDIA H100 80GB HBM3')
        with self.assertRaisesRegex(ValueError,'Only NVIDIA A100 results'):
            self.validate()

    def set_model(self, model):
        self.rewrite_summary(lambda doc: doc['hardware'].update(gpu=model))
        relative='data/results/nvidia-gpu/gpu/dense/20261018_000000/run.yaml'
        metadata=yaml.safe_load((self.bundle/relative).read_text())
        metadata['cuda']['device_name']=model
        (self.bundle/relative).write_text(yaml.safe_dump(metadata))
        manifest=json.loads((self.bundle/'publication.json').read_text())
        manifest['suites'][0]['files'][relative]=digest(self.bundle/relative)
        (self.bundle/'publication.json').write_text(json.dumps(manifest))

    def test_nonstandard_hardware_and_conflicting_revision_rejected(self):
        for field, change in [('hardware', lambda doc: doc['hardware'].update(gpu='NVIDIA H100 80GB HBM3')),
                              ('revision', lambda doc: doc.update(tenferro_commit='c'*40))]:
            with self.subTest(field=field):
                original = {path: path.read_bytes() for path in self.bundle.rglob('*') if path.is_file()}
                self.rewrite_summary(change)
                with self.assertRaises(ValueError):
                    self.validate()
                for path, content in original.items():
                    path.write_bytes(content)


if __name__ == '__main__':
    unittest.main()
