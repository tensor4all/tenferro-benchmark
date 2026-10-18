"""Offline collect/validate/commit/render integration and workflow boundaries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import generate_dashboard as dashboard
from scripts.ci.result_ingestion import plan_updates
from scripts.ci.tests import test_gpu_ingestion as gpu_fixture

ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    def test_accepted_results_history_provenance_and_offline_reproducible_render(self):
        fixture = gpu_fixture.GPUIngestionTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.source.update(repository='tensor4all/tenferro-benchmark',
                              workflow_path='.github/workflows/benchmark-runpod-gpu.yml',
                              head_branch='main', expires_at='2027-01-16T00:00:00Z')
        validated = fixture.validate()
        repo = fixture.root/'repository'
        repo.mkdir()
        def git(*args):
            return subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True,text=True).stdout
        git('init','--initial-branch=main')
        git('config','user.name','Fixture')
        git('config','user.email','fixture@example.invalid')
        receipt,payload = validated[0]
        for relative,data in payload.items():
            path = repo/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            if relative.endswith('.summary.json'):
                document=json.loads(data)
                document['timestamp']='2026-10-17T00:00:00Z'
                document['rows'][0]['median']=0.25
                data=json.dumps(document,indent=2).encode()+b'\n'
            path.write_bytes(data)
        legacy=repo/'result/amd-cpu/cpu/legacy.md'
        legacy.parent.mkdir(parents=True)
        legacy.write_text('# Legacy diagnostic\nNo inferred timestamp.\n')
        git('add','result')
        git('commit','-m','Earlier committed baseline and legacy report')
        updates,decisions=plan_updates(repo,validated)
        self.assertEqual(decisions,{'gpu/dense':'accepted'})
        for relative,data in updates.items():
            (repo/relative).write_bytes(data)
        git('add','result')
        git('commit','-m','Accept exact validated run')
        current=json.loads(payload['result/nvidia-gpu/gpu/dense.summary.json'])
        hardware,previous=dashboard.previous_rows_for('nvidia-gpu','gpu/dense',current,None,repo,'result')
        self.assertEqual(hardware,'NVIDIA A100 80GB PCIe')
        self.assertEqual(previous[0]['median'],0.25)
        before={str(path.relative_to(repo)):path.read_bytes() for path in (repo/'result').rglob('*') if path.is_file()}
        with patch('socket.socket',side_effect=AssertionError('Renderer must not access network')):
            first,second=fixture.root/'site-a',fixture.root/'site-b'
            self.assertEqual(dashboard.generate(repo/'result',first,git_repo=repo,benchmarks_dir=repo/'benchmarks'),0)
            for path in (repo/'result').rglob('*'):
                if path.is_file():os.utime(path,(1234567890,1234567890))
            self.assertEqual(dashboard.generate(repo/'result',second,git_repo=repo,benchmarks_dir=repo/'benchmarks'),0)
        snapshot=lambda root:{str(path.relative_to(root)):path.read_bytes() for path in root.rglob('*') if path.is_file()}
        self.assertEqual(snapshot(first),snapshot(second))
        self.assertEqual(before,{str(path.relative_to(repo)):path.read_bytes() for path in (repo/'result').rglob('*') if path.is_file()})
        page=(first/'nvidia-gpu/gpu/dense.html').read_text()
        self.assertIn('/actions/runs/123/artifacts/456',page)
        self.assertIn('2027-01-16T00:00:00Z',page)
        self.assertIn('Expired artifacts are not replaced',page)
        self.assertTrue((first/'nvidia-gpu/gpu/dense.provenance.json').exists())
        self.assertIn('Legacy report:',(first/'amd-cpu/cpu/legacy.html').read_text())
        self.assertFalse((first/'linux-cpu').exists())
        self.assertFalse((repo/'data/results').exists())
        self.assertFalse((first/'raw').exists())
        (repo/'result/nvidia-gpu/gpu/dense.md').write_text('Corrupted accepted report')
        with self.assertRaisesRegex(ValueError,'differs from provenance'):
            dashboard.generate(repo/'result',fixture.root/'corrupt',git_repo=repo,benchmarks_dir=repo/'benchmarks')
        (repo/'result/nvidia-gpu/gpu/dense.md').write_bytes(payload['result/nvidia-gpu/gpu/dense.md'])
        (repo/'result/nvidia-gpu/gpu/dense.summary.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'differs from provenance'):
            dashboard.generate(repo/'result',fixture.root/'corrupt-summary',git_repo=repo,benchmarks_dir=repo/'benchmarks')

    def test_workflow_permissions_triggers_and_stage_boundaries(self):
        load=lambda name:yaml.load((ROOT/'.github/workflows'/name).read_text(),Loader=yaml.BaseLoader)
        ingestion=load('ingest-results.yml')
        publish=load('publish-results.yml')
        self.assertEqual(ingestion['permissions'],{'actions':'read','contents':'write'})
        self.assertEqual(ingestion['concurrency']['queue'],'max')
        self.assertEqual(publish['permissions']['contents'],'read')
        checkout=publish['jobs']['build']['steps'][0]
        self.assertEqual(checkout['with']['ref'],'main')
        self.assertEqual(checkout['with']['fetch-depth'],'0')
        self.assertNotIn('actions',publish['permissions'])
        self.assertNotIn('workflow_run',publish['on'])
        self.assertEqual(publish['on']['repository_dispatch']['types'],['benchmark-results-ingested'])
        self.assertIn('scripts/generate_dashboard.py',publish['on']['push']['paths'])
        self.assertIn('result/**',publish['on']['push']['paths'])
        text=(ROOT/'.github/workflows/publish-results.yml').read_text()
        for forbidden in ('download-artifact','collect_gpu_pages_results','gh run','_artifacts','_site/raw'):
            self.assertNotIn(forbidden,text)
        for platform in ('mac','linux'):
            collection=load(f'benchmark-nightly-{platform}-cpu.yml')
            self.assertEqual(collection['permissions']['contents'],'read')
            self.assertNotIn('push',collection['on'])
            steps=next(iter(collection['jobs'].values()))['steps']
            bundle=[step for step in steps if 'bundle_cpu_results.py' in step.get('run','')]
            self.assertEqual(len(bundle),1)
            uploads=[step for step in steps if step.get('uses')=='actions/upload-artifact@v4']
            self.assertEqual(uploads[0]['with']['path'],'_cpu_benchmark_artifact/')


if __name__=='__main__':unittest.main()
