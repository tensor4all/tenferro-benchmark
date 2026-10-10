from pathlib import Path
import argparse,subprocess,yaml
parser=argparse.ArgumentParser(description='Generate frozen-workload wheel preparation validation')
parser.add_argument('--tenferro-checkout',type=Path,required=True)
parser.add_argument('--output',type=Path,default=Path('.github/workflows/benchmark-runpod-gpu.yml'))
args=parser.parse_args()
repo=args.tenferro_checkout
base='356a2011ef0f9d2d345ed31201fbe838995329a2'
tested='853fabf55fdfe7d55960c0427c850d34db21dcb0'
raw=subprocess.check_output(['git','show',base+':.github/workflows/runpod-gpu-execute.yml'],cwd=repo,text=True)
p=yaml.safe_load(raw)
p.pop(True,None);p.pop('on',None)
p['name']='benchmark-runpod-gpu'
p['on']={'workflow_dispatch':{'inputs':{'arm':{'type':'choice','options':['baseline','candidate'],'default':'candidate','required':True},'prepare_only':{'type':'boolean','default':True,'description':'Verify frozen artifacts without creating a Pod'}}}}
p['concurrency']={'group':'runpod-preexpanded-wheels','cancel-in-progress':False}
p['permissions']={'contents':'read','actions':'read','checks':'read'}
p['env']['ARCHIVE_WORKSPACE_ROOT']='/home/runner/work/tenferro-rs/tenferro-rs'
p['env']['EXPERIMENT_ARM']='${{ inputs.arm }}'
archive='cuda-pjrt-archive-v16-ci-ubuntu22.04-rust1.99.0-cuda12080-ptx12.8-e16267f9d88e1b65381854d1891b6de2af60188c15d1197d586be2b24dcf891d'
replacements={'${{ inputs.tenferro_ref }}':tested,'${{ inputs.target_head_sha }}':tested,'${{ inputs.target_base_sha }}':base,'${{ inputs.pr_number }}':'0','${{ inputs.runtime_artifact_prefix }}':'gpu-runtime-38034449617','${{ inputs.archive_cache_key }}':archive,'${{ inputs.archive_artifact_name }}':archive,"${{ inputs.keep_failed_pods || 'false' }}":'false'}
def replace(v):
 if isinstance(v,str):
  for a,b in replacements.items():v=v.replace(a,b)
  return v
 if isinstance(v,dict):return {k:replace(x) for k,x in v.items()}
 if isinstance(v,list):return [replace(x) for x in v]
 return v
p=replace(p)
j=p['jobs']['start-runpod'];j['needs']='verify-artifacts';j['if']="${{ !inputs.prepare_only }}"
j['steps']=[s for s in j['steps'] if s['name'] not in ['Require trusted parent context','Revalidate queued PR before provisioning','Record intentionally retained startup failures']]
for job in p['jobs'].values():
 for s in job['steps']:
  if s.get('uses','').startswith('actions/checkout') and s['name']!='Checkout tenferro-rs':
   s['with']={'repository':'tensor4all/tenferro-rs','ref':base,'persist-credentials':False}
  if s.get('uses','').startswith('actions/download-artifact'):
   s['with'].update({'repository':'tensor4all/tenferro-rs','run-id':38034449617,'github-token':'${{ github.token }}'})
  if s['name']=='Checkout tenferro-rs':s['with']['persist-credentials']=False
  if s['name']=='Create GitHub App token':
   s['with'].pop('client-id');s['with']['app-id']='${{ secrets.GH_APP_ID }}'
   s['with']['permission-organization-self-hosted-runners']='write'
  if s['name']=='Define runner label':s['run']=s['run'].replace('runner_label=runpod-','runner_label=runpod-benchmark-')
  if s['name']=='Delete pod if runner startup failed':s['if']="failure() && steps.create_pod.outputs.pod_id != ''"
  if s['name']=='Delete RunPod pod':s.pop('if',None)
  if s['name']=='Decide whether the paid path runs':s['run']='echo "run_paid_path=true" >> "$GITHUB_OUTPUT"\necho "decision=run" >> "$GITHUB_OUTPUT"\n'
  if s['name']=='Provision cheapest compatible RunPod pod':
   s['env']['PROVISION_RUNNER_GROUP_ID']="${{ vars.RUNPOD_RUNNER_GROUP_ID || '4' }}"
   s['run']=s['run'].replace('--pod-env "SMOKE_MIN_RUNTIME_VERSION=', '--pod-env "EXPERIMENT_ARM=${EXPERIMENT_ARM}" \\\n  --pod-env "SMOKE_MIN_RUNTIME_VERSION=')
# Fix GPU model and available driver tier for both arms before creating any pod.
config={'name':'Freeze GPU model and price ceiling','env':{'RUNPOD_API_KEY':'${{ secrets.RUNPOD_API_KEY }}'},'run':'''python3 - <<'CONFIG'
import json
from pathlib import Path
from scripts.ci.runpod_pricing import fetch_gpu_offers, http_transport, GRAPHQL_URL
p=Path('scripts/ci/runpod_config.json')
c=json.loads(p.read_text())
gpu='NVIDIA A40'
offers=fetch_gpu_offers([gpu], min_vram_gb=8, transport=http_transport(GRAPHQL_URL))
if not offers or offers[0].price_per_hr > 0.60:
    raise SystemExit('A40 unavailable below $0.60/hr; no Pod created')
c['gpu_tiers']=[{'name':'experiment','gpu_type_ids':[gpu]}]
c['allowed_cuda_versions']=['13.0','12.9','12.8']
c['max_price_candidates']=1
c['max_provision_attempts']=1
c['same_tier_retries']=0
p.write_text(json.dumps(c))
print(f'A40 offered at ${offers[0].price_per_hr}/hr')
CONFIG
'''}
j['steps'].insert(next(i for i,s in enumerate(j['steps']) if s.get('id')=='create_pod'),config)
p['jobs']={'verify-artifacts':{'name':'Verify frozen workload artifacts','runs-on':'ubuntu-24.04','timeout-minutes':5,'steps':[{'name':'Check source run and artifact inventory','env':{'GH_TOKEN':'${{ github.token }}'},'run':'''set -euo pipefail
gh api repos/tensor4all/tenferro-rs/actions/runs/38034449617/artifacts > /tmp/frozen-artifacts.json
python3 - <<'CHECK'
import json
x=json.load(open('/tmp/frozen-artifacts.json'))['artifacts']
for stem in ('gpu-runtime-38034449617-common','gpu-runtime-38034449617-cuda12.8','''+repr(archive+'-transfer')+'''):
    for part in range(5):
        matches=[a for a in x if a['name']==f'{stem}-part{part:02}']
        assert len(matches)==1 and not matches[0]['expired'], (stem,part)
        print(matches[0]['id'], matches[0]['name'], matches[0]['digest'])
CHECK
'''},{'name':'Verify cross-repository artifact access','uses':'actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c','with':{'repository':'tensor4all/tenferro-rs','run-id':38034449617,'name':'runpod-stage-cost-38034449617-1','path':'verified-cost-record','github-token':'${{ github.token }}'}},{'name':'Save frozen artifact manifest','uses':'actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a','with':{'name':'bootstrap-frozen-manifest','path':'/tmp/frozen-artifacts.json'}}]},**p['jobs']}

import shlex
wrapper='''from scripts.ci import runpod_provision
original_payload = runpod_provision.build_pod_payload

def scoped_payload(*args, **kwargs):
    payload = original_payload(*args, **kwargs)
    payload['dataCenterIds'] = ['CA-MTL-1']
    payload['allowedCudaVersions'] = ['12.8']
    return payload

runpod_provision.build_pod_payload = scoped_payload
raise SystemExit(runpod_provision.main())
'''
for step in p['jobs']['start-runpod']['steps']:
 if step['name']=='Freeze GPU model and price ceiling':
  script=step['run'].replace('fetch_gpu_offers, http_transport, GRAPHQL_URL','parse_gpu_offers, http_transport, GRAPHQL_URL')
  query='{ gpuTypes(input:{id:"NVIDIA A40"}) { id memoryInGb secureCloud securePrice lowestPrice(input:{gpuCount:1,secureCloud:true,dataCenterId:"CA-MTL-1"}) { stockStatus uninterruptablePrice } } }'
  script=script.replace('offers=fetch_gpu_offers([gpu], min_vram_gb=8, transport=http_transport(GRAPHQL_URL))', 'query='+repr(query)+'\nstatus,body=http_transport(GRAPHQL_URL)(json.dumps({"query":query}).encode())\nif status != 200: raise SystemExit("Regional price unavailable; no Pod created")\noffers=parse_gpu_offers(body,[gpu],min_vram_gb=8)')
  step['run']=script.replace("['13.0','12.9','12.8']", "['12.8']")
 if step.get('id')=='create_pod':
  script=step['run']
  step['run']=script.replace('python3 -m scripts.ci.runpod_provision', 'python3 -c '+shlex.quote(wrapper))

# Host identity evidence is taken from the already bounded pre-delete read.
cleanup=p['jobs']['cleanup-runpod']['steps']
for step in cleanup:
 if step['name']=='Read pod record for cost reporting':
  step['run']=step['run'].replace('https://rest.runpod.io/v1/pods/${POD_ID}', 'https://rest.runpod.io/v1/pods/${POD_ID}?includeMachine=true')
index=next(i for i,s in enumerate(cleanup) if s['name']=='Report paid GPU CI cost by stage')
cleanup.insert(index,{'name':'Record placement after deletion','if':"always() && steps.delete_pod.outputs.deleted_at != ''",'continue-on-error':True,'run':'''python3 - <<'PLACEMENT'
import json
from pathlib import Path
pod=json.loads(Path('/tmp/runpod-pod.json').read_text())
machine=pod.get('machine') or {}
record={'pod_id':pod.get('id'),'machine_id':pod.get('machineId'),'data_center_id':machine.get('dataCenterId'),'cpu_type':machine.get('cpuType')}
print(json.dumps(record))
Path('/tmp/runpod-placement.json').write_text(json.dumps(record,indent=2))
PLACEMENT
'''})
cleanup.append({'name':'Save placement observations','if':"always() && steps.delete_pod.outputs.deleted_at != ''",'continue-on-error':True,'uses':'actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a','with':{'name':'runpod-placement-${{ github.run_id }}','path':'/tmp/runpod-placement.json','if-no-files-found':'warn'}})

candidate='64e0692e'
source=yaml.safe_load(subprocess.check_output(['git','show',candidate+':.github/workflows/runpod-gpu-execute.yml'],cwd=repo,text=True))
new_pjrt=next(s['run'] for s in source['jobs']['run-gpu-tests']['steps'] if s['name']=='Run OpenXLA PJRT E2E tests from archive')
for step in p['jobs']['run-gpu-tests']['steps']:
 if step['name']=='Run OpenXLA PJRT E2E tests from archive':step['run']=new_pjrt
 if step['name']=='Download staged execution payload':
  step['with'].update({'pattern':"${{ inputs.arm == 'candidate' && 'prepared-wheels-part*' || 'gpu-runtime-38034449617-common-part*' }}",'run-id':"${{ inputs.arm == 'candidate' && github.run_id || '38034449617' }}",'repository':"${{ inputs.arm == 'candidate' && github.repository || 'tensor4all/tenferro-rs' }}"})
 if step['name']=='Check machine':step['run']+='\nlscpu\ncat /proc/loadavg\n'
 if step['name']=='GPU test complete':step['run']+='\ncat /proc/loadavg\n'
p['jobs']['start-runpod']['needs']='prepare-wheels'
steps=[{'name':'Checkout validation helpers','uses':'actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0','with':{'persist-credentials':False}},
 {'name':'Checkout candidate producer','uses':'actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0','with':{'repository':'tensor4all/tenferro-rs','ref':subprocess.check_output(['git','rev-parse',candidate],cwd=repo,text=True).strip(),'path':'source','persist-credentials':False}},
 {'name':'Download frozen complete wheel payload','uses':'actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c','with':{'repository':'tensor4all/tenferro-rs','run-id':38034449617,'pattern':'gpu-runtime-38034449617-common-part*','merge-multiple':True,'path':'input','github-token':'${{ github.token }}'}},
 {'name':'Run actual producer transform and verify every file','run':'python3 scripts/runpod-preexpanded-wheels/prepare.py --artifacts input --producer source/scripts/ci/prepare_gpu_execution_payload.sh --output prepared'},
 {'name':'Verify tools in the unchanged official runner','run':'''docker run --rm --user root --entrypoint bash \
  -v "$PWD/prepared/decoded:/payload:ro" \
  ghcr.io/actions/actions-runner:2.337.0@sha256:e5496277be5d09bc968b3d64911b74e219ac4a3f2edce956a3ecf9271bea1ef4 -euc '
  /payload/bin/cargo --version
  /payload/bin/cargo-nextest --version
  for ptxas in /payload/wheels-unpacked/*/nvidia/cuda_nvcc/bin/ptxas; do "$ptxas" --version; done
' '''},
 {'name':'Retain complete file and byte verification','uses':'actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a','with':{'name':'preexpanded-wheel-verification','path':'prepared/verification.json','if-no-files-found':'error'}}]
for index in range(5):
 steps.append({'name':f'Upload prepared common part {index:02}','uses':'actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a','with':{'name':f'prepared-wheels-part{index:02}','path':f'prepared/{index:02}/','compression-level':0,'if-no-files-found':'error','retention-days':7}})
p['jobs']['prepare-wheels']={'name':'Prepare and verify complete extracted wheels (no GPU)','needs':'verify-artifacts','runs-on':'ubuntu-24.04','timeout-minutes':20,'steps':steps}

class D(yaml.SafeDumper):pass
def represent(d,s):
 if '\n' in s:s='\n'.join(x.rstrip() for x in s.split('\n'))
 return d.represent_scalar('tag:yaml.org,2002:str',s,style='|' if '\n' in s else None)
D.add_representer(str,represent)
out=args.output
out.write_text('# Full-workload validation of pre-extracted wheels on the official runner.\n# Frozen production controller '+base+'\n'+yaml.dump(p,Dumper=D,sort_keys=False,width=110))
assert 'inputs.keep_failed_pods' not in out.read_text()
