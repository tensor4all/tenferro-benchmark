"""One bounded Pod probe: capture system/container logs before deletion."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from scripts.ci.runpod_client import CreateRequest, _http_transport, build_pod_payload, create_pod
from scripts.ci.runpod_http import delete_pod, pod_request, request_once
from scripts.ci.runpod_provision import _runner_minter, _runner_online_checker

REVISION = "9fcab989978a67d20bd9e253f8c774e4ab88bc67"
IMAGE = "ghcr.io/actions/actions-runner:2.337.0@sha256:e5496277be5d09bc968b3d64911b74e219ac4a3f2edce956a3ecf9271bea1ef4"


def read_logs(url, key):
    """Read a bounded SSE snapshot, including both container and system logs."""
    request = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {key}", "Accept": "text/event-stream",
    })
    chunks = []
    deadline = time.monotonic() + 3
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            while time.monotonic() < deadline and sum(map(len, chunks)) < 65536:
                chunk = response.read1(4096)
                if not chunk:
                    break
                chunks.append(chunk)
    except urllib.error.HTTPError as error:
        return [], error.code
    except (OSError, TimeoutError):
        pass
    events = []
    for line in b"".join(chunks).decode("utf-8", errors="replace").splitlines():
        if line.startswith("data:"):
            try:
                event = json.loads(line[5:])
                if isinstance(event, dict):
                    events.append({k: event.get(k) for k in ("ts", "source", "line")})
            except ValueError:
                pass
    return events, 200 if chunks else 0


def main():
    key = os.environ["RUNPOD_API_KEY"]
    config = json.loads(Path("scripts/ci/runpod_config.json").read_text())
    config.update(min_cuda_version="12.6", max_provision_attempts=1, same_tier_retries=0)
    directory = Path("_runpod_startup_probe")
    directory.mkdir(exist_ok=True)
    gh_token = os.environ.get("PROBE_GH_TOKEN")
    label = "runpod-startup-probe-" + os.environ.get("GITHUB_RUN_ID", "local")
    jit = ""
    runner_online = None
    if gh_token:
        org = os.environ["PROBE_ORG"]
        request = urllib.request.Request(
            f"https://api.github.com/orgs/{org}/actions/runners?per_page=100",
            headers={"Authorization": f"Bearer {gh_token}", "Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            registry = json.load(response)
        print(f"Runner registry: total_count={registry.get('total_count')}, first_page={len(registry.get('runners', []))}", flush=True)
        jit = _runner_minter(gh_token, org, int(os.environ["PROBE_RUNNER_GROUP_ID"]))(label)
        runner_online = _runner_online_checker(gh_token, org)
    # Same smoke helper as the failed run. Only a single-use JIT configuration
    # reaches the Pod; long-lived credentials stay on the hosted side.
    with urllib.request.urlopen(
        f"https://raw.githubusercontent.com/tensor4all/tenferro-rs/{REVISION}/scripts/ci/cuda_smoke_test.py",
        timeout=20,
    ) as response:
        smoke = response.read().decode()
    startup = '''set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then
  exec sudo -E bash -lc "$BASH_EXECUTION_STRING"
fi
trap 'printf "PROBE_EXIT:%s\\n" "$?"' EXIT
export DEBIAN_FRONTEND=noninteractive RUNNER_ALLOW_RUNASROOT=1
echo PROBE_STAGE:tools
command -v curl tar gzip python3
echo PROBE_STAGE:driver
nvidia-smi
echo PROBE_STAGE:smoke
cat > /tmp/cuda_smoke_test.py <<'SMOKE_SOURCE'
''' + smoke + '''
SMOKE_SOURCE
env -u RUNNER_JIT_CONFIG python3 /tmp/cuda_smoke_test.py --min-runtime-version 12.6 --full-runtime-version 12.8 --min-vram-gb 8
echo PROBE_STAGE:runner-binary
cd /home/runner
./bin/Runner.Listener --version
echo PROBE_STAGE:github-network
curl -fsSI --connect-timeout 5 --max-time 10 https://api.github.com/ > /dev/null
echo PROBE_READY
sleep 300
'''
    if jit:
        startup = startup.replace("echo PROBE_READY\nsleep 300", 'echo PROBE_STAGE:jit-registration\nexec ./run.sh --jitconfig "${RUNNER_JIT_CONFIG}"')
    payload = build_pod_payload(config, IMAGE, startup, jit, ["NVIDIA RTX A4000"], {
        "NVIDIA_VISIBLE_DEVICES": "all", "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
    })
    if not jit:
        payload["env"].pop("RUNNER_JIT_CONFIG", None)
    result = create_pod(config, [CreateRequest("startup-probe", json.dumps(payload).encode(), ("NVIDIA RTX A4000",))],
                        transport=_http_transport(config["api_url"], key), secrets=(key, jit))
    pod_id = result.pod_id
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        stream.write(f"pod_id={pod_id}\n")
    print(f"Diagnostic Pod {pod_id} created; at most 240s of observation.", flush=True)
    seen = set()
    ready = False
    events_path = directory / "events.jsonl"
    try:
        deadline = time.monotonic() + 240
        with events_path.open("a", buffering=1) as stream:
            while time.monotonic() < deadline:
                response = request_once(pod_request(config["api_url"], key, pod_id, "GET"), timeout=5)
                observation = {"observed_at": time.time(), "http_status": response.status}
                if response.status == 200:
                    pod = json.loads(response.body)
                    observation.update(status=pod.get("status"), runtime_present=pod.get("runtime") is not None,
                                       cuda_version=pod.get("cudaVersion"))
                line = json.dumps(observation).replace(key, "[REDACTED]")
                print(line, flush=True)
                stream.write(line + "\n")
                events, status = read_logs(config["api_url"] + f"/{pod_id}/logs?tail=200", key)
                stream.write(json.dumps({"logs_http_status": status}) + "\n")
                for event in events:
                    line = json.dumps(event).replace(key, "[REDACTED]")
                    if jit:
                        line = line.replace(jit, "[REDACTED JIT]")
                    if line not in seen:
                        seen.add(line)
                        print(line, flush=True)
                        stream.write(line + "\n")
                    ready |= event.get("line") == "PROBE_READY"
                if runner_online:
                    ready = runner_online(label, deadline=deadline)
                    print(f"Runner registry online={ready}", flush=True)
                if ready:
                    break
                time.sleep(8)
        print(f"Pre-registration probe ready={ready}", flush=True)
    finally:
        if not delete_pod(config["api_url"], key, pod_id, deadline_seconds=60):
            raise RuntimeError("Diagnostic Pod deletion not confirmed")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
