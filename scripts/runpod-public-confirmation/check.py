"""Check generated lifecycle wiring, unchanged workloads, and draft dispatch guard."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import yaml


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tenferro-checkout", type=Path, required=True)
args = parser.parse_args()
protocol = json.loads(Path("result/nvidia-gpu/ci/public-runner-confirmation-draft.json").read_text())
workflow = yaml.safe_load(Path(".github/workflows/benchmark-runpod-gpu.yml").read_text())
original = yaml.safe_load(subprocess.check_output([
    "git", "show", protocol["candidate_source"] + ":.github/workflows/runpod-gpu-execute.yml",
], cwd=args.tenferro_checkout, text=True))
for job in workflow["jobs"].values():
    for step in job["steps"]:
        if "run" in step:
            result = subprocess.run(["bash", "-n"], input=step["run"], text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(step["name"] + ": " + result.stderr)
source_steps = {step["name"]: step for step in original["jobs"]["run-gpu-tests"]["steps"]}
actual_steps = {step["name"]: step for step in workflow["jobs"]["run-gpu-tests"]["steps"]}
for name in ["Run CUDA tests from archive", "Run CUDA tutorial artifact", "Run OpenXLA PJRT E2E tests from archive",
             "Install staged execution payload", "Install selected CUDA SDK", "Verify loaded NVRTC version"]:
    assert actual_steps[name]["run"] == source_steps[name]["run"], name
assert workflow["on"]["workflow_dispatch"]["inputs"]["prepare_only"]["default"] is True
assert workflow["jobs"]["start-runpod"]["if"] == "${{ !inputs.prepare_only }}"
assert workflow["jobs"]["start-runpod"]["needs"] == "verify-artifacts"
guard = next(step["run"] for step in workflow["jobs"]["verify-artifacts"]["steps"] if step["name"] == "Keep draft comparisons free")
for prepare_only, expected in [("true", 0), ("false", int(protocol["status"].startswith("DRAFT")))]:
    result = subprocess.run(["bash", "-euc", guard], env=dict(os.environ, PREPARE_ONLY=prepare_only), capture_output=True, text=True)
    assert result.returncode == expected, result.stdout + result.stderr
cleanup = workflow["jobs"]["cleanup-runpod"]
assert cleanup["if"] == "always() && needs.start-runpod.outputs.pod_id != ''"
names = [step["name"] for step in cleanup["steps"]]
assert names.index("Delete RunPod pod") < names.index("Record placement after deletion")
assert "if" not in cleanup["steps"][names.index("Delete RunPod pod")]
for name in ["start-runpod", "run-gpu-tests"]:
    assert workflow["jobs"][name].get("permissions", workflow["permissions"]).get("actions") != "write"
for step in actual_steps.values():
    assert not step.get("uses", "").startswith("actions/cache/save")
assert workflow["jobs"]["setup-watchdog"]["steps"][-1]["run"] == original["jobs"]["setup-watchdog"]["steps"][-1]["run"]
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    bound = dict(protocol, harness_commit="0" * 40, status="DRAFT fixture")
    (root / "protocol.json").write_text(json.dumps(bound))
    command = ["python3", "scripts/runpod-public-confirmation/run-comparison.py"]
    env = dict(os.environ, RUNPOD_COMPARISON_DIR=directory)
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "decision is pending" in result.stderr
    assert not (root / "state.json").exists()
    bound["status"] = "accepted fixture only"
    (root / "protocol.json").write_text(json.dumps(bound))
    original_state = '{"runs":[{"run_id":123}],"status":"running"}\n'
    (root / "state.json").write_text(original_state)
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "Campaign state exists" in result.stderr
    assert (root / "state.json").read_text() == original_state
print("Shell syntax, unchanged execution, draft guard, cache roles and mandatory cleanup passed")
