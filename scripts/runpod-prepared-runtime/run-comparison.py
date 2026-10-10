import datetime,json,os,re,statistics,subprocess,time
from compare import evaluate
from pathlib import Path
ROOT=Path(os.environ.get('RUNPOD_COMPARISON_DIR','/tmp/runpod-prepared-confirmation'))
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
REPO='tensor4all/tenferro-benchmark'
BRANCH='experiment/runpod-prepared-comparison'
assert re.fullmatch(r'[0-9a-f]{40}',PROTOCOL.get('harness_commit') or ''), 'Bind the committed harness before starting'
assert re.fullmatch(r'ghcr.io/tensor4all/tenferro-ci-prepared-runner@sha256:[0-9a-f]{64}',PROTOCOL.get('candidate_image') or ''), 'Bind the public immutable candidate image before starting'
assert not (ROOT/'state.json').exists(), 'Campaign state exists; inspect/resume its existing run, never overwrite or restart it'
assert PROTOCOL['attempts']==1
STATE={'pid':os.getpid(),'status':'starting','runs':[]}
def save():
    (ROOT/'state.json').write_text(json.dumps(STATE,indent=2)+'\n')
def command(*args,timeout=120):
    result=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr[:1000])
    return result.stdout
def gh(*args):return command('gh',*args)
def clean(s):
    s=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',s)
    return re.sub(r'\^\[\[[0-9;]*[A-Za-z]','',s)
def observation(path,run):
    cost=json.loads((path/'cost/runpod-stage-cost.json').read_text())
    placement=json.loads((path/'placement/runpod-placement.json').read_text())
    log=clean((path/'run.log').read_text())
    driver=re.search(r'0, NVIDIA A40, [0-9]+ MiB, ([0-9.]+)',log)
    runtime=re.search(r'Selected CUDA runtime: ([0-9.]+) \(full',log)
    cpu=re.search(r'Model name:\s+([^\n]+)',log)
    assert driver and runtime and cpu, 'missing host identity'
    assert re.search(r'285 tests run: 285 passed',log), 'CUDA case-count drift'
    assert re.search(r'3 tests run: 3 passed',log), 'PJRT case-count drift'
    assert 'RUNPOD_TENFERRO_GPU_TEST_OK' in log
    assert cost['gpu_job_conclusion']=='success'
    assert placement['data_center_id']=='CA-MTL-1', placement
    assert cost['tested_ref']==PROTOCOL['tested_ref']
    assert cost['price_per_hour']<=0.60
    assert cost['paid_seconds']>0
    assert runtime.group(1)==PROTOCOL['hardware']['selected_runtime']
    assert cost['gpu_type_id']==PROTOCOL['hardware']['gpu']
    assert 'RunPod delete HTTP status: 204' in log or 'RunPod delete HTTP status: 404' in log
    cases=re.findall(r'PASS\s+\[\s*[0-9.]+s\]\s+\(\s*\d+/\d+\)\s+([^\n]+)',log)
    assert sorted(x.strip() for x in cases)==PROTOCOL['workload']['test_case_ids'], 'exact case identity drift'
    assert 'cuda_tutorial: upload -> session -> download passed' in log

    return {'run_id':run,'driver':driver.group(1),'runtime':runtime.group(1),'cpu':cpu.group(1).strip(),'placement':placement,**cost}
try:
    save()
    for index,arm_label in enumerate(PROTOCOL['order']):
        arm='baseline' if arm_label=='A' else 'candidate'
        spent=sum(r.get('observation',{}).get('estimated_gpu_cost',0) for r in STATE['runs'])
        # Reserve more than the 1-hour lifetime ceiling at $0.60/h before allocation.
        assert spent+0.70<=PROTOCOL['limits']['max_aggregate_gpu_cost_usd'], 'Cost ceiling leaves insufficient room for another bounded Pod'

        remote=gh('api',f'repos/{REPO}/git/ref/heads/{BRANCH}','--jq','.object.sha').strip()
        assert remote==PROTOCOL['harness_commit'], 'harness branch moved; no new Pod allocated'
        STATE['status']='dispatching'
        STATE['next_index']=index
        STATE['next_arm']=arm
        save()
        # A dispatch error is not retried: its outcome may be unknown. Inspect
        # the actual workflow list before deciding whether another is needed.
        output=gh('workflow','run','benchmark-runpod-gpu.yml','-R',REPO,'--ref',BRANCH,'-f',f'arm={arm}','-f','prepare_only=false','-f','candidate_image='+PROTOCOL['candidate_image'])
        match=re.search(r'/actions/runs/(\d+)',output)
        assert match, 'dispatch outcome unknown; inspect repository before retrying'
        run=int(match.group(1))
        entry={'index':index,'arm':arm_label,'run_id':run,'status':'running'}
        STATE['runs'].append(entry)
        STATE['status']='running'
        save()
        print(f'{index+1}/{len(PROTOCOL['order'])} {arm}: https://github.com/{REPO}/actions/runs/{run}',flush=True)
        while True:
            try:
                result=json.loads(gh('run','view',str(run),'-R',REPO,'--json','status,conclusion,jobs'))
            except (RuntimeError,subprocess.TimeoutExpired,json.JSONDecodeError) as error:
                print(f'Observation failed for existing run {run}; keep polling: {error}',flush=True)
                time.sleep(30)
                continue
            entry['workflow_status']=result['status']
            entry['active_steps']=[s['name'] for j in result['jobs'] for s in j.get('steps',[]) if s['status']=='in_progress']
            save()
            if result['status']=='completed':break
            time.sleep(30)
        path=ROOT/str(run);path.mkdir()
        (path/'jobs.json').write_text(json.dumps(result,indent=2)+'\n')
        entry['conclusion']=result['conclusion'];save()
        (path/'run.log').write_text(gh('run','view',str(run),'-R',REPO,'--log'))
        # Failed or missing evidence remains visible and stops further spending;
        # no missing run is silently replaced by another successful sample.
        gh('run','download',str(run),'-R',REPO,'--name',f'runpod-stage-cost-{run}-1','--dir',str(path/'cost'))
        gh('run','download',str(run),'-R',REPO,'--name',f'runpod-placement-{run}','--dir',str(path/'placement'))
        assert result['conclusion']=='success', f'workflow {run} failed; inspect cleanup'
        observed=observation(path,run)
        (path/'observation.json').write_text(json.dumps(observed,indent=2)+'\n')
        entry.update(status='complete',observation=observed);save()
        print(f'{run}: {observed["paid_seconds"]:.3f}s, {observed["driver"]}, machine={observed["placement"]["machine_id"]}',flush=True)
        if index==1:
            aa=[r['observation']['paid_seconds'] for r in STATE['runs']]
            assert max(aa)/min(aa)<=PROTOCOL['validity']['aa_max_min'], 'A/A spread exceeded; INCONCLUSIVE'
    rows=[dict(r['observation'],arm=r['arm']) for r in STATE['runs']]
    decision=evaluate(rows,PROTOCOL)
    STATE.update(status='complete',decision=decision,total_estimated_gpu_cost=sum(r['estimated_gpu_cost'] for r in rows))
    save();print(json.dumps(decision),flush=True)
except Exception as error:
    STATE.update(status='needs_review',error=str(error));save()
    print('Campaign stopped; no automatic replacement: '+str(error),flush=True)
    raise
