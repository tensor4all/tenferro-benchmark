#!/usr/bin/env python3
"""Create and inspect the RunPod network volume that holds the JAX compile cache.

The volume is a standing resource: the weekly GPU job mounts it so XLA's compilation cache
survives between ephemeral pods.  A network volume is bound to one data center and the
binding is immutable, so `create` insists on `--data-center` and refuses to guess, and a
pod that mounts the volume must be created in that same data center -- `pod_data_center`
reads the binding back from the API so callers never have to duplicate it.

Standard library only, and the HTTP policy (bearer auth, retry, cursor pagination) is the
same module the pod watchdog uses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

from scripts.ci.runpod_client import redacted_error_message
from scripts.ci.runpod_http import Cooldown, request_with_retry, retry_delay, retryable

DEFAULT_API_BASE = "https://api.runpod.io/v2"
DEFAULT_CONFIG = "scripts/ci/runpod_config.json"
MIN_SIZE_GB = 10
MAX_SIZE_GB = 4096
#: The scheduler picks an arbitrary host, but a network volume mount is only
#: accepted in the volume's own data center, so the pod request has to name it.
DEFAULT_GPU_TIER = "a100"


def _headers(api_key: str, *, json_body: bool = False) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def _call(request, *, deadline_seconds, transport, monotonic, sleep, cooldown=None,
          attempts=None):
    kwargs = {
        "deadline": monotonic() + deadline_seconds,
        "cooldown": cooldown if cooldown is not None else Cooldown(),
        "monotonic": monotonic,
        "sleep": sleep,
    }
    if attempts is not None:
        kwargs["attempts"] = attempts
    if transport is not None:
        kwargs["transport"] = transport
    return request_with_retry(request, **kwargs)


def pod_api_base(api_url: str) -> str:
    """`.../v2/pods` -> `.../v2`; the volume endpoints hang off the same base."""
    base = api_url.rsplit("/pods", 1)[0]
    return base or DEFAULT_API_BASE


def validate_create_request(*, name, size_gb, data_center) -> None:
    """Reject what the provider would reject, before anything is sent or charged."""
    if not name:
        raise ValueError("a volume name is required")
    if not data_center:
        raise ValueError("a data center is required: a network volume cannot move after creation")
    if not isinstance(size_gb, int) or isinstance(size_gb, bool):
        raise ValueError("size must be an integer number of GB")
    if not MIN_SIZE_GB <= size_gb <= MAX_SIZE_GB:
        raise ValueError(f"size must be an integer between {MIN_SIZE_GB} and {MAX_SIZE_GB} GB")


def _volume_record(volume):
    return {
        "id": volume["id"],
        "name": volume.get("name"),
        "size": volume.get("size"),
        "dataCenter": volume.get("dataCenter"),
    }


def _list_request(api_base, api_key, cursor):
    url = api_base.rstrip("/") + "/network-volumes"
    if cursor is not None:
        url += "?" + urllib.parse.urlencode({"cursor": cursor})
    return urllib.request.Request(url, headers=_headers(api_key))


def list_network_volumes(api_base, api_key, *, deadline_seconds=60.0, max_pages=20,
                         transport=None, monotonic=time.monotonic, sleep=time.sleep):
    """Every network volume the key can see, following the cursor pagination."""
    cooldown = Cooldown()
    volumes, cursor, seen = [], None, set()
    for _ in range(max_pages):
        response = _call(
            _list_request(api_base, api_key, cursor),
            deadline_seconds=deadline_seconds,
            transport=transport,
            monotonic=monotonic,
            sleep=sleep,
            cooldown=cooldown,
        )
        if response.status != 200:
            raise ValueError(f"network volume listing failed (HTTP {response.status})")
        payload = json.loads(response.body)
        if not isinstance(payload, dict) or not isinstance(payload.get("networkVolumes"), list):
            raise ValueError("v2 network volume list requires a networkVolumes array")
        for volume in payload["networkVolumes"]:
            if not isinstance(volume, dict) or not isinstance(volume.get("id"), str):
                raise ValueError("v2 network volume list contains an invalid volume")
            volumes.append(_volume_record(volume))
        pagination = payload.get("pagination")
        if not isinstance(pagination, dict) or not isinstance(
            pagination.get("hasNextPage"), bool
        ):
            raise ValueError("v2 network volume list requires pagination metadata")
        if not pagination["hasNextPage"]:
            return volumes
        cursor = pagination.get("nextCursor")
        if not isinstance(cursor, str) or not cursor or cursor in seen:
            # An unfinished page that claims a next page but carries no usable
            # cursor used to end the listing early -- and the caller then created
            # a duplicate of a volume it had not seen.
            raise ValueError("v2 network volume list has an invalid or repeated cursor")
        seen.add(cursor)
    raise ValueError("network volume pagination did not terminate")


def fetch_network_volume(api_base, api_key, volume_id, *, deadline_seconds=60.0,
                         transport=None, monotonic=time.monotonic, sleep=time.sleep):
    """Read one volume, e.g. to learn the data center its pods must use."""
    if not volume_id:
        raise ValueError("a volume id is required")
    request = urllib.request.Request(
        api_base.rstrip("/") + "/network-volumes/" + urllib.parse.quote(volume_id, safe=""),
        headers=_headers(api_key),
    )
    response = _call(request, deadline_seconds=deadline_seconds, transport=transport,
                     monotonic=monotonic, sleep=sleep)
    if response.status != 200:
        raise ValueError(f"network volume {volume_id} lookup failed (HTTP {response.status})")
    volume = json.loads(response.body)
    if not isinstance(volume, dict) or not isinstance(volume.get("id"), str):
        raise ValueError("network volume lookup returned no id")
    return _volume_record(volume)


def pod_data_center(api_base, api_key, volume_id, **kwargs) -> str:
    """The data center a pod must be created in to mount `volume_id`."""
    data_center = fetch_network_volume(api_base, api_key, volume_id, **kwargs).get("dataCenter")
    if not isinstance(data_center, str) or not data_center:
        raise ValueError(f"network volume {volume_id} reports no data center; refusing to guess")
    return data_center


def list_data_centers(api_base, api_key, *, deadline_seconds=60.0, transport=None,
                      monotonic=time.monotonic, sleep=time.sleep):
    """The data center catalog with per-GPU availability."""
    request = urllib.request.Request(
        api_base.rstrip("/") + "/catalog/datacenters?"
        + urllib.parse.urlencode({"include": "GPU_AVAILABILITY"}),
        headers=_headers(api_key),
    )
    response = _call(request, deadline_seconds=deadline_seconds, transport=transport,
                     monotonic=monotonic, sleep=sleep)
    if response.status != 200:
        raise ValueError(f"data center catalog failed (HTTP {response.status})")
    payload = json.loads(response.body)
    if not isinstance(payload, dict) or not isinstance(payload.get("dataCenters"), list):
        raise ValueError("v2 data center catalog requires a dataCenters array")
    return [entry for entry in payload["dataCenters"] if isinstance(entry, dict)]


def check_data_center(api_base, api_key, data_center, *, required_gpu_type_ids=(),
                      **kwargs) -> str:
    """Refuse a data center that cannot host the volume, and return its canonical spelling.

    The bindings this checks are the ones that cannot be undone afterwards: a volume
    cannot move, so a data center without the benchmark GPU (or without network volume
    support) would be a permanently useless paid resource.
    """
    wanted = data_center.strip().upper()
    centers = list_data_centers(api_base, api_key, **kwargs)
    match = next(
        (entry for entry in centers if str(entry.get("id", "")).upper() == wanted), None
    )
    if match is None:
        known = ", ".join(sorted(str(entry.get("id")) for entry in centers)) or "none"
        raise ValueError(f"{data_center!r} is not an available data center (available: {known})")
    canonical = str(match["id"])
    if not match.get("networkVolumeTypes"):
        raise ValueError(f"{canonical} does not support network volumes")
    if required_gpu_type_ids:
        availability = {
            str(gpu.get("id")): gpu.get("availability")
            for gpu in match.get("gpuAvailability") or []
            if isinstance(gpu, dict)
        }
        # One in-stock benchmark type is enough to place pods here; requiring the
        # whole tier would reject every real data center, which offers a subset.
        offered = [
            gpu_id for gpu_id in required_gpu_type_ids
            if availability.get(gpu_id) not in (None, "NONE")
        ]
        if not offered:
            raise ValueError(
                f"{canonical} offers none of {', '.join(required_gpu_type_ids)}; a volume "
                "created there could never be mounted by the benchmark pods"
            )
    return canonical


def create_network_volume(api_base, api_key, *, name, size_gb, data_center,
                          deadline_seconds=120.0, attempts=4, reconciliation_reads=3,
                          transport=None, monotonic=time.monotonic, sleep=time.sleep):
    """POST the volume and return the created record.

    A lost response is ambiguous: the volume may exist already, or the provider may
    still be creating it. Re-sending the POST would pay for a second volume, so an
    ambiguous attempt is only ever reconciled by reading the listing back; when the
    volume does not appear, the run stops and asks for a human re-check. Only a
    rate-limited attempt, which by definition created nothing, is retried.
    """
    validate_create_request(name=name, size_gb=size_gb, data_center=data_center)
    body = json.dumps({"name": name, "size": size_gb, "dataCenter": data_center}).encode()
    request = urllib.request.Request(
        api_base.rstrip("/") + "/network-volumes",
        data=body,
        method="POST",
        headers=_headers(api_key, json_body=True),
    )
    cooldown = Cooldown()
    response = None
    for attempt in range(1, attempts + 1):
        response = _call(request, deadline_seconds=deadline_seconds, transport=transport,
                         monotonic=monotonic, sleep=sleep, cooldown=cooldown, attempts=1)
        if response.status in (200, 201):
            created = json.loads(response.body)
            if not isinstance(created, dict) or not isinstance(created.get("id"), str):
                raise ValueError("network volume creation returned no id")
            return _volume_record(created)
        if not retryable(response.status):
            raise ValueError(
                f"network volume creation failed (HTTP {response.status}): "
                f"{_diagnostic(response.body, api_key)}"
            )
        if response.status == 429:
            if attempt == attempts:
                raise ValueError(
                    "network volume creation is rate limited (HTTP 429); try again later"
                )
            delay = retry_delay(response.headers, attempt)
            cooldown.until = max(cooldown.until, monotonic() + delay)
            continue
        break  # the request may have reached the provider: do not re-POST
    adopted = None
    for index in range(reconciliation_reads):
        adopted = _select_volume(
            [volume for volume in list_network_volumes(
                api_base, api_key, transport=transport, monotonic=monotonic, sleep=sleep
            ) if volume.get("name") == name],
            name=name, size_gb=size_gb, data_center=data_center,
        )
        if adopted is not None:
            return adopted
        if index + 1 < reconciliation_reads:
            sleep(retry_delay(response.headers, 1))
    raise ValueError(
        f"network volume creation failed (HTTP {response.status if response else 0}) and no "
        "volume with that name is visible yet; the provider may still be creating it, so "
        "re-run `list` before trying again"
    )


def _diagnostic(body: bytes, api_key: str) -> str:
    """A bounded error body with the request credential removed."""
    return redacted_error_message(body, secrets=(api_key,))[:400]


def _mismatches(volume, *, size_gb, data_center) -> list[str]:
    problems = []
    found_center = volume.get("dataCenter")
    if isinstance(found_center, str) and found_center.upper() != data_center.strip().upper():
        problems.append(f"data center {found_center!r} != requested {data_center!r}")
    found_size = volume.get("size")
    if isinstance(found_size, int) and found_size != size_gb:
        problems.append(f"size {found_size} GB != requested {size_gb} GB")
    return problems


def _select_volume(matches, *, name, size_gb, data_center):
    """The single volume that may be reused for this request, or None.

    Every path that reuses a volume -- the idempotent lookup and the adoption of
    a volume left behind by an ambiguous POST -- goes through this check, so a
    reused volume always has the bindings the caller asked for.
    """
    if len(matches) > 1:
        ids = ", ".join(str(volume["id"]) for volume in matches)
        raise ValueError(
            f"{len(matches)} network volumes are named {name!r} ({ids}); a name is not a "
            "unique key, so pick the intended id explicitly and delete the others"
        )
    if not matches:
        return None
    problems = _mismatches(matches[0], size_gb=size_gb, data_center=data_center)
    if problems:
        raise ValueError(
            f"network volume {matches[0]['id']} is named {name!r} but does not match the "
            f"request ({'; '.join(problems)}); its bindings are immutable, so choose "
            "another name or reuse it as it is"
        )
    return matches[0]


def ensure_network_volume(api_base, api_key, *, name, size_gb, data_center, dry_run=False,
                          **kwargs):
    """Idempotent by name, and strict about the bindings that cannot be changed later."""
    validate_create_request(name=name, size_gb=size_gb, data_center=data_center)
    existing = _select_volume(
        [volume for volume in list_network_volumes(api_base, api_key, **kwargs)
         if volume.get("name") == name],
        name=name, size_gb=size_gb, data_center=data_center,
    )
    if existing is not None:
        return {"created": False, "volume": existing,
                "note": "a volume with this name already exists; the id above is the one to use"}
    if dry_run:
        return {"created": False, "volume": None,
                "request": {"name": name, "size": size_gb, "dataCenter": data_center},
                "note": "dry run: nothing was created"}
    return {"created": True, "volume": create_network_volume(
        api_base, api_key, name=name, size_gb=size_gb, data_center=data_center, **kwargs)}


def required_gpu_type_ids(config, tier=DEFAULT_GPU_TIER) -> tuple[str, ...]:
    """The GPU types the benchmark pods ask for, from the reviewed tier table."""
    for entry in config.get("gpu_tiers") or []:
        if isinstance(entry, dict) and entry.get("name") == tier:
            return tuple(str(gpu) for gpu in entry.get("gpu_type_ids") or ())
    return ()


def _api_base_from_config(path, override):
    if override:
        return override
    try:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
        api_url = config.get("api_url") or ""
    except (OSError, ValueError):
        api_url = ""
    return pod_api_base(api_url)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--skip-data-center-check", action="store_true",
                        help="do not consult the data center catalog before creating")
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

    try:
        if args.command == "list":
            print(json.dumps(list_network_volumes(api_base, api_key),
                             ensure_ascii=False, indent=1))
            return 0

        validate_create_request(name=args.name, size_gb=args.size_gb,
                                data_center=args.data_center)
        data_center = args.data_center.strip()
        if not args.skip_data_center_check:
            try:
                with open(args.config, encoding="utf-8") as handle:
                    config = json.load(handle)
            except (OSError, ValueError):
                config = {}
            data_center = check_data_center(
                api_base, api_key, data_center,
                required_gpu_type_ids=required_gpu_type_ids(config),
            )
        result = ensure_network_volume(
            api_base, api_key, name=args.name, size_gb=args.size_gb,
            data_center=data_center, dry_run=args.dry_run,
        )
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=1))
    volume = result.get("volume") or {}
    if volume.get("id"):
        print(f"\nSet the repository variable RUNPOD_JAX_CACHE_VOLUME_ID={volume['id']} "
              f"(data center {volume.get('dataCenter')}).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
