"""Check experiment scripts and that an unpublished image cannot allocate Pods."""
import os
import json
import tempfile
from pathlib import Path
import subprocess
import yaml

workflow=yaml.safe_load(Path('.github/workflows/benchmark-runpod-gpu.yml').read_text())
for name,job in workflow['jobs'].items():
    for step in job.get('steps',[]):
        if 'run' not in step:continue
        result=subprocess.run(['bash','-n'],input=step['run'],text=True,capture_output=True)
        if result.returncode:raise RuntimeError(f'{name}/{step["name"]}: {result.stderr}')
preflight=workflow['jobs']['verify-artifacts']['steps'][0]['run']
for prepare,image,success in [('true','',True),('false','',False),
                               ('false','ghcr.io/tensor4all/tenferro-ci-prepared-runner:latest',False),
                               ('false','https://example.invalid/image',False)]:
    result=subprocess.run(['bash','-euc',preflight],env=dict(os.environ,PREPARE_ONLY=prepare,CANDIDATE_IMAGE=image),capture_output=True,text=True)
    if (result.returncode==0)!=success:raise RuntimeError(result.stdout+result.stderr)
    print(f'prepare_only={prepare}, image={image!r}: expected outcome verified')
start=workflow['jobs']['start-runpod']
assert start['needs']=='verify-artifacts'
assert '!inputs.prepare_only' in start['if']
assert workflow['on']['workflow_dispatch']['inputs']['prepare_only']['default'] is True
assert 'packages' not in workflow['permissions']
print('All workflow shell scripts parse; default and missing-image paths cannot allocate Pods')

# A lost local observer must never replace an already-dispatched paid run.
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    protocol=json.loads(Path('result/nvidia-gpu/ci/runpod-prepared-runtime-protocol.json').read_text())
    protocol.update(harness_commit='0'*40, candidate_image='ghcr.io/tensor4all/tenferro-ci-prepared-runner@sha256:'+'0'*64)
    (root/'protocol.json').write_text(json.dumps(protocol))
    original='{"status":"running","runs":[{"run_id":123}]}\n'
    (root/'state.json').write_text(original)
    result=subprocess.run(['python3','scripts/runpod-prepared-runtime/run-comparison.py'],env=dict(os.environ,RUNPOD_COMPARISON_DIR=directory),capture_output=True,text=True)
    assert result.returncode!=0 and 'Campaign state exists' in result.stderr
    assert (root/'state.json').read_text()==original
    print('Existing campaign state is preserved and cannot trigger a replacement dispatch')
