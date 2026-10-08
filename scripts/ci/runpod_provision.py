#!/usr/bin/env python3
"""Create one ephemeral RunPod pod for the benchmark GitHub runner.

The long-lived RunPod API key is used only by the GitHub-hosted provisioning
job.  The pod receives only the single-use GitHub JIT runner configuration.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("RunPod config must be an object")
    return value


def build_payload(
    config: dict[str, Any],
    *,
    image: str,
    startup_script: str,
    jit_config: str,
    label: str,
    run_id: str,
) -> dict[str, Any]:
    return {
        "cloudType": config["cloud_type"],
        "computeType": config["compute_type"],
        "allowedCudaVersions": config["allowed_cuda_versions"],
        "name": f"tenferro-benchmark-gpu-{run_id}",
        "imageName": image,
        "gpuTypeIds": config["gpu_type_ids"],
        "gpuTypePriority": config["gpu_type_priority"],
        "gpuCount": config["gpu_count"],
        "containerDiskInGb": config["container_disk_gb"],
        "volumeInGb": config["volume_gb"],
        "volumeMountPath": config["volume_mount_path"],
        "interruptible": False,
        "ports": [],
        "dockerEntrypoint": ["/bin/bash", "-lc"],
        "dockerStartCmd": [
            "printf %s "
            + base64.b64encode(startup_script.encode()).decode()
            + " | base64 -d | bash"
        ],
        "env": {
            "RUNNER_JIT_CONFIG": jit_config,
            "RUNNER_LABEL": label,
            "RUNNER_ALLOW_RUNASROOT": "1",
        },
    }


def create_pod(config: dict[str, Any], payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        str(config["api_url"]),
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "tenferro-benchmark-runpod/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"RunPod create failed with HTTP {error.code}: {detail}") from error
    if not isinstance(body, dict) or not isinstance(body.get("id"), str):
        raise RuntimeError("RunPod create response did not contain a pod id")
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--startup-script", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    jit_config = os.environ.get("RUNNER_JIT_CONFIG")
    label = os.environ.get("RUNNER_LABEL")
    if not api_key or not jit_config or not label:
        raise RuntimeError("RUNPOD_API_KEY, RUNNER_JIT_CONFIG and RUNNER_LABEL are required")

    config = load_config(args.config)
    payload = build_payload(
        config,
        image=args.image,
        startup_script=args.startup_script.read_text(encoding="utf-8"),
        jit_config=jit_config,
        label=label,
        run_id=os.environ.get("GITHUB_RUN_ID", "local"),
    )
    # Never print the payload: it contains the single-use JIT configuration.
    result = create_pod(config, payload, api_key)
    pod_id = result["id"]
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"pod_id={pod_id}\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"### RunPod benchmark pod\n\n- Pod ID: `{pod_id}`\n- Runner label: `{label}`\n")
    print(f"Created RunPod benchmark pod {pod_id}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"RunPod provisioning failed: {error}", file=sys.stderr)
        raise SystemExit(1)
