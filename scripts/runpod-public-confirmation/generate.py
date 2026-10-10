"""Generate a draft bootstrap comparison from the two declared source revisions."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

import yaml


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tenferro-checkout", type=Path, required=True)
args = parser.parse_args()
protocol = json.loads(Path("result/nvidia-gpu/ci/public-runner-confirmation-draft.json").read_text())
current = protocol["candidate_source"]
baseline = protocol["baseline_source"]
run_id = protocol["artifact_run"]
archive = protocol["archive_key"]


def source(revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=args.tenferro_checkout, text=True)


workflow = yaml.safe_load(source(current, ".github/workflows/runpod-gpu-execute.yml"))
historical = yaml.safe_load(source(baseline, ".github/workflows/runpod-gpu-execute.yml"))
workflow.pop(True, None)
workflow.pop("on", None)
workflow["name"] = "benchmark-runpod-gpu"
workflow["on"] = {"workflow_dispatch": {"inputs": {
    "arm": {"type": "choice", "options": ["baseline", "candidate"], "default": "baseline"},
    "prepare_only": {"type": "boolean", "default": True, "description": "Free artifact and native-image preflight; do not create a GPU Pod"},
}}}
workflow["permissions"] = {"contents": "read", "actions": "read", "checks": "read"}
workflow["concurrency"] = {"group": "runpod-public-confirmation", "cancel-in-progress": False}
workflow["env"].update({
    "EXPERIMENT_ARM": "${{ inputs.arm }}",
    "ARCHIVE_WORKSPACE_ROOT": "/home/runner/work/tenferro-rs/tenferro-rs",
    "RUNPOD_IMAGE": "${{ inputs.arm == 'baseline' && '" + protocol["baseline_image"] + "' || '" + protocol["candidate_image"] + "' }}",
})

replacements = {
    "${{ inputs.tenferro_ref }}": current,
    "${{ inputs.target_head_sha }}": current,
    "${{ inputs.target_base_sha }}": current,
    "${{ inputs.pr_number }}": "0",
    "${{ inputs.runtime_artifact_prefix }}": f"gpu-runtime-{run_id}",
    "${{ inputs.archive_cache_key }}": archive,
    "${{ inputs.archive_artifact_name }}": archive,
    "${{ inputs.keep_failed_pods || 'false' }}": "false",
}


def replace(value):
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
    elif isinstance(value, list):
        value = [replace(item) for item in value]
    elif isinstance(value, dict):
        value = {key: replace(item) for key, item in value.items()}
    return value


workflow = replace(workflow)
start = workflow["jobs"]["start-runpod"]
start["needs"] = "verify-artifacts"
start["if"] = "${{ !inputs.prepare_only }}"
start["steps"] = [step for step in start["steps"] if step["name"] not in {
    "Require trusted parent context", "Revalidate queued PR before provisioning",
    "Record intentionally retained startup failures",
}]
old_steps = {step["name"]: step for step in historical["jobs"]["start-runpod"]["steps"]}
old_tools = historical["jobs"]["run-gpu-tests"]["steps"][0]["run"]
for name, job in workflow["jobs"].items():
    for step in job["steps"]:
        if step.get("uses", "").startswith("actions/checkout"):
            if step["name"] != "Checkout tenferro-rs":
                step["with"] = {"repository": "tensor4all/tenferro-rs", "ref": current, "persist-credentials": False}
            else:
                step["with"]["persist-credentials"] = False
        if step.get("uses", "").startswith("actions/download-artifact"):
            step["with"].update({"repository": "tensor4all/tenferro-rs", "run-id": run_id, "github-token": "${{ github.token }}"})
        if step["name"] == "Decide whether the paid path runs":
            step["run"] = 'echo "run_paid_path=true" >> "$GITHUB_OUTPUT"\necho "decision=run" >> "$GITHUB_OUTPUT"\n'
        if step["name"] == "Create GitHub App token":
            step["with"].pop("client-id")
            step["with"]["app-id"] = "${{ secrets.GH_APP_ID }}"
            step["with"]["permission-organization-self-hosted-runners"] = "write"
        if step["name"] == "Define runner label":
            step["run"] = step["run"].replace("runner_label=runpod-", "runner_label=runpod-benchmark-")
        if step["name"] == "Delete pod if runner startup failed":
            step["if"] = "failure() && steps.create_pod.outputs.pod_id != ''"
        if step["name"] == "Delete RunPod pod":
            step.pop("if", None)
        if step["name"] == "Provision cheapest compatible RunPod pod":
            step["env"]["PROVISION_RUNNER_GROUP_ID"] = "${{ vars.RUNPOD_RUNNER_GROUP_ID || '4' }}"
            old = replace(old_steps[step["name"]]["run"])
            old = old.replace("cat scripts/ci/cuda_smoke_test.py", "cat historical-ci/scripts/ci/cuda_smoke_test.py")
            step["run"] = 'if [ "$EXPERIMENT_ARM" = baseline ]; then\n' + old + '\nelse\n' + step["run"] + '\nfi\n'
            wrapper = '''from scripts.ci import runpod_provision
original = runpod_provision.build_pod_payload
def scoped(*args, **kwargs):
    payload = original(*args, **kwargs)
    payload['dataCenterIds'] = ['CA-MTL-1']
    payload['allowedCudaVersions'] = ['12.8']
    return payload
runpod_provision.build_pod_payload = scoped
raise SystemExit(runpod_provision.main())
'''
            step["run"] = step["run"].replace("python3 -m scripts.ci.runpod_provision", "python3 -c " + shlex.quote(wrapper))
        if step["name"] == "Install pod-side execution dependencies":
            step["run"] = 'if [ "$EXPERIMENT_ARM" = baseline ]; then\n' + old_tools + '\nelse\n' + step["run"] + '\nfi\n'
        if step["name"] == "Check machine":
            step["run"] += '\nlscpu\ncat /proc/loadavg\ndpkg-query -W cuda-nvrtc-12-8\ntest "$(dpkg-query -W -f=\'${Version}\' cuda-nvrtc-12-8)" = 12.8.93-1\n'
        if step["name"] == "GPU test complete":
            step["run"] += "\ncat /proc/loadavg\n"

checkout = next(step for step in start["steps"] if step["name"] == "Checkout trusted RunPod client")
start["steps"].insert(start["steps"].index(checkout) + 1, {
    "name": "Checkout historical smoke installer", "uses": checkout["uses"],
    "with": {"repository": "tensor4all/tenferro-rs", "ref": baseline, "path": "historical-ci", "persist-credentials": False},
})
freeze = {
    "name": "Freeze placement and bootstrap poll interval",
    "env": {"RUNPOD_API_KEY": "${{ secrets.RUNPOD_API_KEY }}"},
    "run": '''python3 - <<'CONFIG'
import json, os
from pathlib import Path
from scripts.ci.runpod_pricing import parse_gpu_offers, http_transport, GRAPHQL_URL
query='{ gpuTypes(input:{id:"NVIDIA A40"}) { id memoryInGb secureCloud securePrice lowestPrice(input:{gpuCount:1,secureCloud:true,dataCenterId:"CA-MTL-1"}) { stockStatus uninterruptablePrice } } }'
status, body = http_transport(GRAPHQL_URL)(json.dumps({'query':query}).encode())
if status != 200: raise SystemExit('Regional price unavailable; no Pod created')
offers = parse_gpu_offers(body, ['NVIDIA A40'], min_vram_gb=8)
if not offers or offers[0].price_per_hr > 0.60:
    raise SystemExit('A40 unavailable below $0.60/hr; no Pod created')
path=Path('scripts/ci/runpod_config.json')
config=json.loads(path.read_text())
config.update(gpu_tiers=[{'name':'experiment','gpu_type_ids':['NVIDIA A40']}],
              allowed_cuda_versions=['12.8'],max_price_candidates=1,max_provision_attempts=1,
              same_tier_retries=0,startup_poll_seconds=10 if os.environ['EXPERIMENT_ARM']=='baseline' else 2)
path.write_text(json.dumps(config))
print('Fixed A40 price:', offers[0].price_per_hr)
CONFIG
''',
}
start["steps"].insert(next(i for i, step in enumerate(start["steps"]) if step["name"] == "Create GitHub App token"), freeze)

cleanup = workflow["jobs"]["cleanup-runpod"]
cleanup["if"] = "always() && needs.start-runpod.outputs.pod_id != ''"
for step in cleanup["steps"]:
    if step["name"] == "Read pod record for cost reporting":
        step["run"] = step["run"].replace("https://rest.runpod.io/v1/pods/${POD_ID}", "https://rest.runpod.io/v1/pods/${POD_ID}?includeMachine=true")
cleanup["steps"].extend([
    {"name": "Record placement after deletion", "if": "always() && steps.delete_pod.outputs.deleted_at != ''", "continue-on-error": True,
     "run": '''python3 - <<'PY'
import json
from pathlib import Path
pod=json.loads(Path('/tmp/runpod-pod.json').read_text())
machine=pod.get('machine') or {}
value={'pod_id':pod.get('id'),'machine_id':pod.get('machineId'),
       'data_center_id':machine.get('dataCenterId'),'cpu_type':machine.get('cpuType')}
Path('/tmp/runpod-placement.json').write_text(json.dumps(value,indent=2))
print(json.dumps(value))
PY
'''},
    {"name": "Save placement observations", "if": "always() && steps.delete_pod.outputs.deleted_at != ''", "continue-on-error": True,
     "uses": "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
     "with": {"name": "runpod-placement-${{ github.run_id }}", "path": "/tmp/runpod-placement.json", "if-no-files-found": "warn"}},
])

verify_script = f'''set -euo pipefail
gh api repos/tensor4all/tenferro-rs/actions/runs/{run_id} > /tmp/frozen-run.json
gh api repos/tensor4all/tenferro-rs/actions/runs/{run_id}/artifacts > /tmp/frozen-artifacts.json
python3 - <<'PY'
import json
run=json.load(open('/tmp/frozen-run.json'))
assert run['head_sha']=={current!r} and run['conclusion']=='success'
artifacts=json.load(open('/tmp/frozen-artifacts.json'))['artifacts']
for stem in ({f'gpu-runtime-{run_id}-common'!r}, {f'gpu-runtime-{run_id}-cuda12.8'!r}, {archive+'-transfer'!r}):
    for part in range(5):
        found=[x for x in artifacts if x['name']==f'{{stem}}-part{{part:02}}']
        assert len(found)==1 and not found[0]['expired'], (stem,part)
        print(found[0]['id'], found[0]['name'], found[0]['digest'])
PY
'''
workflow["jobs"] = {"verify-artifacts": {
    "name": "Verify frozen workload artifacts", "runs-on": "ubuntu-24.04", "timeout-minutes": 10,
    "steps": [
        {"name": "Checkout comparison definition", "uses": "actions/checkout@v5", "with": {"persist-credentials": False}},
        {"name": "Keep draft comparisons free", "env": {"PREPARE_ONLY": "${{ inputs.prepare_only }}"},
         "run": '''python3 - <<'PY'
import json,os
from pathlib import Path
protocol=json.loads(Path('result/nvidia-gpu/ci/public-runner-confirmation-draft.json').read_text())
if os.environ['PREPARE_ONLY']!='true' and protocol['status'].startswith('DRAFT'):
    raise SystemExit('The comparison design is still a draft; paid dispatch is disabled')
print('Preflight mode:', os.environ['PREPARE_ONLY'])
PY
'''},
        {"name": "Check source run and artifact inventory", "env": {"GH_TOKEN": "${{ github.token }}"}, "run": verify_script},
        {"name": "Verify cross-repository artifact access", "uses": "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
         "with": {"repository": "tensor4all/tenferro-rs", "run-id": run_id, "name": f"runpod-stage-cost-{run_id}-1", "path": "verified-cost-record", "github-token": "${{ github.token }}"}},
        {"name": "Save frozen artifact manifest", "uses": "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
         "with": {"name": "public-runner-frozen-manifest", "path": "/tmp/frozen-artifacts.json"}},
    ],
}, **workflow["jobs"]}


class Dumper(yaml.SafeDumper):
    pass


def represent(dumper, value):
    if "\n" in value:
        value = "\n".join(line.rstrip() for line in value.split("\n"))
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|" if "\n" in value else None)


Dumper.add_representer(str, represent)
Path(".github/workflows/benchmark-runpod-gpu.yml").write_text(
    "# Generated by scripts/runpod-public-confirmation/generate.py; no GPU on default dispatch.\n"
    + yaml.dump(workflow, Dumper=Dumper, sort_keys=False, width=110)
)
