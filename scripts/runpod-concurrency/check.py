"""Check experiment wiring before allocating any Pod."""
import argparse,json,os,subprocess,tempfile
from pathlib import Path
import yaml
parser=argparse.ArgumentParser();parser.add_argument('--tenferro-checkout',type=Path,required=True);args=parser.parse_args()
p=json.loads(Path('result/nvidia-gpu/ci/runpod-concurrency-protocol.json').read_text())
w=yaml.safe_load(Path('.github/workflows/benchmark-runpod-gpu.yml').read_text())
o=yaml.safe_load(subprocess.check_output(['git','show',p['candidate_source']+':.github/workflows/runpod-gpu-execute.yml'],cwd=args.tenferro_checkout,text=True))
for j in w['jobs'].values():
 for step in j['steps']:
  if 'run' in step:subprocess.run(['bash','-n'],input=step['run'],text=True,check=True)
assert w['jobs']['start-runpod']['if']=='${{ !inputs.prepare_only }}'
assert w['jobs']['start-runpod']['needs']=='verify-artifacts'
assert w['on']['workflow_dispatch']['inputs']['prepare_only']['default'] is True
steps={s['name']:s for s in w['jobs']['run-gpu-tests']['steps']}
original={s['name']:s for s in o['jobs']['run-gpu-tests']['steps']}
for name in ('Run CUDA tutorial artifact','Run OpenXLA PJRT E2E tests from archive','Install staged execution payload','Install selected CUDA SDK','Verify loaded NVRTC version'):
 assert steps[name]['run']==original[name]['run'],name
assert '--test-threads "${NEXTEST_TEST_THREADS}"' in steps['Run CUDA tests from archive']['run']
assert steps['Run OpenXLA PJRT E2E tests from archive']['env']['NEXTEST_TEST_THREADS']=='1'
assert steps['Checkout tenferro-rs']['with']['ref']==p['candidate_source']
cleanup=w['jobs']['cleanup-runpod'];names=[s['name'] for s in cleanup['steps']]
assert names.index('Delete RunPod pod')<names.index('Report paid GPU CI cost by stage')
assert 'if' not in next(s for s in cleanup['steps'] if s['name']=='Delete RunPod pod')
assert w['jobs']['setup-watchdog']['steps'][-1]['run']==o['jobs']['setup-watchdog']['steps'][-1]['run']
assert 'pod_metadata' in w['jobs']['start-runpod']['outputs']
for name in ('start-runpod','run-gpu-tests'):
 assert w['jobs'][name].get('permissions',w['permissions']).get('actions')!='write'
# Check both exit statuses: the monitoring trap must never hide a test failure.
prefix=steps['Run CUDA tests from archive']['run'].split('trap finish_memory EXIT',1)[0]+'trap finish_memory EXIT\n'
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);stub=root/'nvidia-smi';stub.write_text('#!/bin/sh\necho "1024, 46068"\n');stub.chmod(0o755)
 for status in (0,1):
  result=subprocess.run(['bash','-c',prefix.replace('/tmp/cuda-memory.csv',str(root/'memory.csv'))+f'sleep 0.1\nexit {status}\n'],env=dict(os.environ,PATH=directory+':'+os.environ['PATH']),capture_output=True,text=True)
  assert result.returncode==status,(result.stdout,result.stderr)
  assert 'CUDA_MEMORY_PEAK_MIB=1024' in result.stdout
print('Shell syntax, pinned workload, serial PJRT, metadata fallback, cleanup, watchdog and memory trap passed.')
