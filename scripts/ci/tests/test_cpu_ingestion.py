import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.ci.bundle_cpu_results import package_cpu
from scripts.ci.cpu_ingestion_adapter import build_source, identities
from scripts.ci.result_ingestion import digest, validate_bundle


class CPUIngestionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/'data/results/amd-cpu/cpu/fft/20261018_000000'
        self.run.mkdir(parents=True)
        self.metadata = {'target_profile': 'amd-cpu', 'suite_id': 'cpu/fft',
                         'timestamp': '2026-10-18T00:00:00+00:00',
                         'tenferro_rs': {'commit': 'b'*40, 'features':['blas-mkl']}, 'blas':{'implementation':'mkl'}, 'python_backends':{'pytorch':{'provider':'mkl'}}, 'harness': {'commit': 'a'*40},
                         'environment': {'hostname': 'fixture', 'cpu': 'Fixture CPU', 'env':{'RAYON_NUM_THREADS':'1'}}}
        (self.run/'run.yaml').write_text(yaml.safe_dump(self.metadata))
        for thread in (1,4):
            meta = json.loads(json.dumps(self.metadata))
            meta['environment']['env']['RAYON_NUM_THREADS']=str(thread)
            (self.run/f'run_t{thread}.yaml').write_text(yaml.safe_dump(meta))
        self.csv = self.run/'cpu_fft_t1_fixture.csv'
        self.csv.write_text('suite,benchmark,dtype,threads,shape,backend,median_ms,iqr_ms,status,notes\n'
                            'cpu/fft,fft,c64,1,1d_n16,tenferro-fft-executor-cached,1.25,0.1,ok,cached\n'
                            'cpu/fft,fft,c64,4,1d_n16,tenferro-fft-executor-cached,0.75,0.1,ok,cached\n')
        report = self.root/'result/amd-cpu/cpu/fft.md'
        report.parent.mkdir(parents=True)
        report.write_text('# FFT\nRaw data: `data/results/amd-cpu/cpu/fft/20261018_000000`.\n')
        descriptor = {'metadata': str((self.run/'run.yaml').relative_to(self.root)),
                      'raw_files': [str(self.csv.relative_to(self.root))],
                      'thread_metadata':[str((self.run/f'run_t{thread}.yaml').relative_to(self.root)) for thread in (1,4)]}
        document = build_source(self.root, descriptor, 'cpu/fft', 'amd-cpu', {'head_sha':'a'*40})
        inventory = self.root/'inventory.json'
        rows = sorted(identities(document))
        inventory.write_text(json.dumps({'schema_version':1, 'threads':[1,4], 'suites':{'cpu/fft':{
            'row_count':len(rows), 'identities_sha256':hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()}}}))
        self.inventory_patch = patch('scripts.ci.cpu_ingestion_adapter.SELECTION_PATH', inventory)
        self.inventory_patch.start()
        self.addCleanup(self.inventory_patch.stop)
        self.bundle = self.root/'bundle'
        self.source = {'head_sha':'a'*40, 'run_id':123, 'run_attempt':1, 'artifact_id':456}

    def package(self):
        return package_cpu(self.root, 'amd-cpu', 'a'*40, self.bundle, suites=('fft',))

    def validate(self):
        return validate_bundle(self.bundle, 'amd-cpu', self.source)

    def rewrite(self, relative, content):
        (self.bundle/relative).write_bytes(content)
        path = self.bundle/'ingestion.json'
        manifest = json.loads(path.read_text())
        manifest['suites'][0]['files'][relative] = digest(self.bundle/relative)
        path.write_text(json.dumps(manifest))

    def test_collection_and_ingestion_rebuild_raw_rows_and_preserve_report(self):
        self.package()
        before = self.csv.read_bytes()
        validated = self.validate()
        receipt,payload = validated[0]
        document = json.loads(payload['result/amd-cpu/cpu/fft.summary.json'])
        self.assertEqual([row['median'] for row in document['rows']], [1.25,0.75])
        self.assertEqual(document['hardware']['threads'],[1,4])
        self.assertEqual(self.csv.read_bytes(),before)
        self.assertEqual(payload['result/amd-cpu/cpu/fft.md'],(self.root/'result/amd-cpu/cpu/fft.md').read_bytes())
        self.assertEqual(receipt['source'],self.source)

    def test_deleted_raw_row_even_with_updated_digests_is_rejected(self):
        manifest = self.package()
        relative = manifest['suites'][0]['sources'][0]['raw_files'][0]
        data = (self.bundle/relative).read_text().splitlines()
        self.rewrite(relative, ('\n'.join(data[:-1])+'\n').encode())
        with self.assertRaisesRegex(ValueError,'thread coverage|selection|raw threads'):
            self.validate()

    def test_forged_summary_median_is_rejected_even_with_updated_digests(self):
        manifest = self.package()
        entry = manifest['suites'][0]
        paths = ['result/amd-cpu/cpu/fft.summary.json', entry['run_directory']+'/report.summary.json']
        for relative in paths:
            document = json.loads((self.bundle/relative).read_text())
            document['rows'][0]['median'] = 999
            self.rewrite(relative,json.dumps(document).encode())
        with self.assertRaisesRegex(ValueError,'differs from raw'):
            self.validate()

    def test_missing_declared_workload_fails_collection(self):
        self.csv.write_text(self.csv.read_text().splitlines()[0]+'\n')
        with self.assertRaisesRegex(ValueError,'No CPU raw measurements|raw threads'):
            self.package()

    def test_mismatched_pytorch_provider_cannot_publish(self):
        self.metadata['python_backends']['pytorch']['provider']='openblas'
        (self.run/'run.yaml').write_text(yaml.safe_dump(self.metadata))
        with self.assertRaisesRegex(ValueError,'MKL-backed PyTorch'):
            self.package()

    def test_mac_profile_uses_the_same_offline_path(self):
        import shutil
        shutil.copytree(self.root/'data/results/amd-cpu', self.root/'data/results/mac-cpu')
        target = self.root/'result/mac-cpu/cpu/fft.md'
        target.parent.mkdir(parents=True)
        target.write_text((self.root/'result/amd-cpu/cpu/fft.md').read_text().replace('amd-cpu','mac-cpu'))
        for path in (self.root/'data/results/mac-cpu').rglob('*.yaml'):
            meta = yaml.safe_load(path.read_text())
            meta['target_profile']='mac-cpu'
            meta['tenferro_rs']['features']=['blas-accelerate']
            meta['blas']['implementation']='accelerate'
            path.write_text(yaml.safe_dump(meta))
        package_cpu(self.root,'mac-cpu','a'*40,self.bundle,suites=('fft',))
        self.assertEqual(validate_bundle(self.bundle,'mac-cpu',self.source)[0][0]['target_profile'],'mac-cpu')

    def test_wrong_source_commit_fails_collection(self):
        self.metadata['harness']['commit']='c'*40
        (self.run/'run.yaml').write_text(yaml.safe_dump(self.metadata))
        with self.assertRaisesRegex(ValueError,'harness commit'):
            self.package()


if __name__=='__main__':
    unittest.main()
