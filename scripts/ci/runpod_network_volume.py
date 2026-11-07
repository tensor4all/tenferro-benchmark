#!/usr/bin/env python3
"""Create and inspect the RunPod network volume that holds the JAX compile cache.

The volume is a standing resource: the weekly GPU job mounts it so XLA's compilation cache
survives between ephemeral pods.  A network volume is bound to one data center and the binding is
immutable, so pods that need the cache have to be created in that data center -- the `create`
subcommand therefore insists on `--data-center` and refuses to guess.

Standard library only, and the HTTP policy (bearer auth, retry, cursor pagination) is the same
module the pod watchdog uses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

from scripts.ci.runpod_http import Cooldown, request_with_retry

DEFAULT_API_BASE = "https://api.runpod.io/v2"
MIN_SIZE_GB = 10
MAX_SIZE_GB = 4096


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}


def list_network_volumes(api_base, api_key, *, deadline_seconds=60.0, max_pages=20,
                         transport=None, monotonic=time.monotonic, sleep=time.sleep):
    """Every network volume the key can see, following the cursor pagination."""
    deadline = monotonic() + deadline_seconds
    cooldown = Cooldown()
    url = api_base.rstrip("/") + "/network-volumes"
    volumes, cursor, seen = [], None, set()
    for _ in range(max_pages):
        page_url = url if cursor is None else url + "?" + urllib.parse.urlencode({"cursor": cursor})
        request = urllib.request.Request(page_url, headers=_headers(api_key))
        kwargs = {"deadline": deadline, "cooldown": cooldown, "monotonic": monotonic, "sleep": sleep}
        if transport is not None:
            kwargs["transport"] = transport
        response = request_with_retry(request, **kwargs)
        if response.status != 200:
            raise ValueError(f"network volume listing failed (HTTP {response.status})")
        payload = json.loads(response.body)
        if not isinstance(payload, dict) or not isinstance(payload.get("networkVolumes"), list):
            raise ValueError("v2 network volume list requires a networkVolumes array")
        for volume in payload["networkVolumes"]:
            if not isinstance(volume, dict) or not isinstance(volume.get("id"), str):
                raise ValueError("v2 network volume list contains an invalid volume")
            volumes.append({
                "id": volume["id"],
                "name": volume.get("name"),
                "size": volume.get("size"),
                "dataCenter": volume.get("dataCenter"),
            })
        pagination = payload.get("pagination") or {}
        cursor = pagination.get("nextCursor") if pagination.get("hasNextPage") else None
        if not cursor:
            return volumes
        if cursor in seen:
            raise ValueError("network volume pagination cursor repeated")
        seen.add(cursor)
    raise ValueError("network volume pagination did not terminate")


def create_network_volume(api_base, api_key, *, name, size_gb, data_center,
                          deadline_seconds=120.0, transport=None,
                          monotonic=time.monotonic, sleep=time.sleep):
    """POST the volume and return the created record.  `dataCenter` is required and immutable."""
    if not name:
        raise ValueError("a volume name is required")
    if not data_center:
        raise ValueError("a data center is required: a network volume cannot move after creation")
    if not isinstance(size_gb, int) or not MIN_SIZE_GB <= size_gb <= MAX_SIZE_GB:
        raise ValueError(f"size must be an integer between {MIN_SIZE_GB} and {MAX_SIZE_GB} GB")
    body = json.dumps({"name": name, "size": size_gb, "dataCenter": data_center}).encode()
    request = urllib.request.Request(
        api_base.rstrip("/") + "/network-volumes",
        data=body,
        method="POST",
        headers={**_headers(api_key), "Content-Type": "application/json"},
    )
    kwargs = {"deadline": monotonic() + deadline_seconds, "cooldown": Cooldown(),
              "monotonic": monotonic, "sleep": sleep}
    if transport is not None:
        kwargs["transport"] = transport
    response = request_with_retry(request, **kwargs)
    if response.status not in (200, 201):
        raise ValueError(f"network volume creation failed (HTTP {response.status}): "
                         f"{response.body[:400].decode('utf-8', 'replace')}")
    created = json.loads(response.body)
    if not isinstance(created, dict) or not isinstance(created.get("id"), str):
        raise ValueError("network volume creation returned no id")
    return created


def ensure_network_volume(api_base, api_key, *, name, size_gb, data_center, dry_run=False, **kwargs):
    """Idempotent: a volume with this name is reused, never duplicated."""
    existing = [v for v in list_network_volumes(api_base, api_key, **kwargs) if v.get("name") == name]
    if existing:
        return {"created": False, "volume": existing[0],
                "note": "a volume with this name already exists; the id above is the one to use"}
    if dry_run:
        return {"created": False, "volume": None,
                "request": {"name": name, "size": size_gb, "dataCenter": data_center},
                "note": "dry run: nothing was created"}
    return {"created": True, "volume": create_network_volume(
        api_base, api_key, name=name, size_gb=size_gb, data_center=data_center, **kwargs)}


def _api_base_from_config(path, override):
    if override:
        return override
    try:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
        api_url = config.get("api_url") or ""
    except (OSError, ValueError):
        api_url = ""
    # runpod_config.json points at .../v2/pods; the volume endpoints hang off the same base.
    return api_url.rsplit("/pods", 1)[0] or DEFAULT_API_BASE


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--config", default="scripts/ci/runpod_config.json")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="print every network volume as JSON")

    create = sub.add_parser("create", help="create the volume, or return the existing one")
    create.add_argument("--name", default="jax-compile-cache")
    create.add_argument("--size-gb", type=int, default=10)
    create.add_argument("--data-center", required=True,
                        help="required: a network volume is pinned to one data center")
    create.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY is not set", file=sys.stderr)
        return 2
    api_base = _api_base_from_config(args.config, args.api_base)

    if args.command == "list":
        print(json.dumps(list_network_volumes(api_base, api_key), ensure_ascii=False, indent=1))
        return 0

    result = ensure_network_volume(
        api_base, api_key, name=args.name, size_gb=args.size_gb,
        data_center=args.data_center, dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=1))
    volume = result.get("volume") or {}
    if volume.get("id"):
        print(f"\nSet the repository variable RUNPOD_JAX_CACHE_VOLUME_ID={volume['id']} "
              f"(data center {volume.get('dataCenter')}).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
