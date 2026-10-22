#!/usr/bin/env python3
"""Read-only A100 capacity diagnostics; never create pods or register runners."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from scripts.ci.runpod_contract import (
    ContractError,
    configured_min_cuda_version,
    configured_gpu_tiers,
    validate_min_cuda_version,
    validate_gpu_type_ids,
)
from scripts.ci.runpod_http import Cooldown, request_with_retry
from scripts.ci.runpod_pricing import PricingError, candidate_plan


def catalog_url(config, *, cuda_filter):
    params = {
        "include": "AVAILABILITY",
        "product": "POD",
        "cloud": config["cloud_type"],
        "count": config["gpu_count"],
    }
    if cuda_filter:
        params["minCudaVersion"] = configured_min_cuda_version(config)
    return config["catalog_url"] + "?" + urllib.parse.urlencode(params)


def inspect_catalog(body, config, tiers):
    """Retain only public catalog fields for the configured A100 IDs."""
    payload = json.loads(body)
    if not isinstance(payload, dict) or not isinstance(payload.get("gpus"), list):
        raise ValueError("catalog response requires a gpus array")
    eligible = [gpu for _, ids in tiers for gpu in ids]
    entries = {
        item["id"]: item for item in payload["gpus"]
        if isinstance(item, dict) and item.get("id") in eligible
    }
    rows = []
    for gpu_id in eligible:
        entry = entries.get(gpu_id, {})
        prices = entry.get("price")
        rows.append({
            "id": gpu_id,
            "present": gpu_id in entries,
            "secure": entry.get("secure"),
            "availability": entry.get("availability"),
            "memory_gb": entry.get("memory"),
            "secure_price_per_hr": prices.get("secure") if isinstance(prices, dict) else None,
            "cuda_versions": entry.get("cudaVersions"),
        })
    logs = io.StringIO()
    with contextlib.redirect_stdout(logs):
        plan = candidate_plan(config, tiers, transport=lambda: (200, body))
    return {"gpus": rows, "candidate_plan": plan, "selection_log": logs.getvalue()}


def diagnose(config, key, *, fetch=request_with_retry):
    tiers = [(name, ids) for name, ids in configured_gpu_tiers(config) if name == "a100"]
    if not tiers:
        raise ValueError("configuration has no a100 tier")
    configured_min_cuda_version(config)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "checks": {},
        "errors": [],
    }
    # Shared cooldown respects Retry-After across these sequential reads.
    cooldown = Cooldown()
    payloads = {}
    checks = [
        ("openapi", config["openapi_url"]),
        ("production", catalog_url(config, cuda_filter=True)),
        ("without_cuda_filter", catalog_url(config, cuda_filter=False)),
    ]
    for name, url in checks:
        request = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "User-Agent": "tenferro-runpod-diagnostics/1",
        })
        response = fetch(request, deadline=time.monotonic() + 60, cooldown=cooldown)
        check = {"url": url, "http_status": response.status}
        report["checks"][name] = check
        if response.status != 200:
            report["errors"].append(f"{name}: HTTP {response.status}")
            continue
        try:
            payloads[name] = json.loads(response.body)
            if name != "openapi":
                check.update(inspect_catalog(response.body, config, tiers))
        except (ValueError, TypeError, PricingError):
            report["errors"].append(f"{name}: invalid JSON or catalog format")
    if "openapi" in payloads and "without_cuda_filter" in payloads:
        try:
            validate_gpu_type_ids(payloads["openapi"], tiers[0][1], payloads["without_cuda_filter"])
            validate_min_cuda_version(payloads["openapi"], configured_min_cuda_version(config))
            report["contract"] = "passed (A100 IDs and CUDA syntax)"
        except (ContractError, KeyError, TypeError, AttributeError):
            report["contract"] = "failed"
            report["errors"].append("A100 catalog/OpenAPI contract validation failed")
    return report


def summary(report):
    lines = [
        "# RunPod A100 diagnostics", "",
        "Read-only snapshot; no pods rented. Catalog availability does not guarantee creation.",
        "CUDA-unfiltered stock is diagnostic only; production still requires its configured CUDA versions.",
        "", f"UTC: {report['timestamp_utc']}", "",
        f"Contract: {report.get('contract', 'not checked')}", "",
    ]
    for name, check in report["checks"].items():
        lines.extend([f"## {name}: HTTP {check['http_status']}", ""])
        for row in check.get("gpus", []):
            # JSON avoids interpreting catalog strings as Markdown/Actions commands.
            lines.append("    " + json.dumps(row, ensure_ascii=True))
        if "selection_log" in check:
            lines.extend(["", "Candidate selection:", ""])
            lines.extend("    " + line for line in check["selection_log"].splitlines())
        lines.append("")
    lines.extend(f"- {error}" for error in report["errors"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("runpod_config.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        (args.output_dir / "summary.md").write_text("RUNPOD_API_KEY is required.\n")
        return 1
    report = diagnose(json.loads(args.config.read_text()), key)
    # Defense in depth: neither API error bodies nor headers are archived.
    encoded = json.dumps(report, indent=2).replace(key, "[REDACTED]")
    rendered = summary(report).replace(key, "[REDACTED]")
    (args.output_dir / "diagnostics.json").write_text(encoded + "\n")
    (args.output_dir / "summary.md").write_text(rendered)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as stream:
            stream.write(rendered)
    print("RunPod diagnostics saved; see summary.md and diagnostics.json.")
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
