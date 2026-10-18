#!/usr/bin/env python3
"""Standard-library-only RunPod HTTP policy, also embedded in the pod watchdog."""
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path


@dataclass
class Cooldown:
    """A process-local endpoint cooldown shared by pricing and pod observations."""
    until: float = 0.0


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes


def retry_after(headers, *, wall_time=time.time):
    value = next((v for k, v in headers.items() if k.lower() == 'retry-after'), None)
    if value is None:
        return None
    try:
        delay = float(value)
    except (ValueError, TypeError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            delay = date.timestamp() - wall_time()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, delay) if math.isfinite(delay) else None


def retryable(status):
    return status in (0, 408, 429, 500, 502, 503, 504)


def retry_delay(headers, attempt, *, base=5.0, cap=60.0, wall_time=time.time):
    advertised = retry_after(headers, wall_time=wall_time)
    backoff = min(cap, base * 2 ** min(attempt - 1, 10))
    backoff = min(cap, backoff * random.uniform(1.0, 1.2))
    # Never shorten the provider's minimum, even when it exceeds our backoff cap.
    return max(advertised or 0.0, backoff)


def request_once(request, *, timeout):
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Response(response.status, dict(response.headers), response.read())
    except urllib.error.HTTPError as error:
        with error:
            return Response(error.code, dict(error.headers or {}), error.read())
    except (OSError, http.client.HTTPException):
        # Do not print URLs, response bodies, or exception strings containing credentials.
        return Response(0, {}, b'')


def request_with_retry(request, *, deadline, attempts=5, timeout=20.0,
                       monotonic=time.monotonic, sleep=time.sleep,
                       transport=request_once):
    response = Response(0, {}, b'')
    for attempt in range(1, attempts + 1):
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        response = transport(request, timeout=min(timeout, remaining))
        if not retryable(response.status) or attempt == attempts:
            break
        delay = retry_delay(response.headers, attempt)
        if delay >= deadline - monotonic():
            break
        print(f'RunPod HTTP {response.status}; retry {attempt}/{attempts} after {delay:.1f}s')
        sleep(delay)
    return response


def pod_request(api_url, api_key, pod_id, method):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', pod_id):
        raise ValueError('invalid pod ID')
    return urllib.request.Request(
        f'{api_url.rstrip("/")}/{pod_id}', method=method,
        headers={'Authorization': f'Bearer {api_key}', 'Accept': 'application/json',
                 'User-Agent': 'tenferro-benchmark-runpod/1'},
    )


def report_leak(pod_id):
    message = (f'Could not confirm deletion of RunPod pod {pod_id}. '
               'It may still be billing. Check and delete it in the RunPod dashboard.')
    print(f'::error::{message}', file=sys.stderr)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a', encoding='utf-8') as stream:
            stream.write(f'\n### RunPod cleanup requires attention\n\n{message}\n')


def delete_pod(api_url, api_key, pod_id, *, deadline_seconds=180.0, attempts=5,
               monotonic=time.monotonic, sleep=time.sleep, transport=request_once):
    deadline = monotonic() + deadline_seconds
    response = request_with_retry(
        pod_request(api_url, api_key, pod_id, 'DELETE'), deadline=deadline,
        attempts=attempts, monotonic=monotonic, sleep=sleep, transport=transport,
    )
    print(f'RunPod delete pod={pod_id} HTTP {response.status}')
    if 200 <= response.status < 300 or response.status == 404:
        return True
    # A lost DELETE response can mean the deletion succeeded. Do not issue a
    # verification GET after 429/503 etc.: Retry-After applies to that too.
    if response.status == 0 and monotonic() < deadline:
        check = request_with_retry(
            pod_request(api_url, api_key, pod_id, 'GET'), deadline=deadline,
            attempts=attempts, monotonic=monotonic, sleep=sleep, transport=transport,
        )
        if check.status == 404:
            return True
    report_leak(pod_id)
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('delete', 'get'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--api-url')
    parser.add_argument('--delay', type=float, default=0)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text()) if args.config else {}
        api_url = args.api_url or config.get('api_url')
        key = os.environ.get('RUNPOD_API_KEY')
        pod_id = os.environ.get('POD_ID') or os.environ.get('RUNPOD_POD_ID')
        if not api_url or not key or not pod_id:
            raise ValueError('API URL, RUNPOD_API_KEY and POD_ID/RUNPOD_POD_ID are required')
        pod_request(api_url, key, pod_id, 'GET')  # validate before sleeping
        if args.delay:
            time.sleep(args.delay)
        if args.action == 'delete':
            return 0 if delete_pod(api_url, key, pod_id) else 1
        response = request_with_retry(pod_request(api_url, key, pod_id, 'GET'),
                                      deadline=time.monotonic() + 60)
        return 0 if 200 <= response.status < 300 else 1
    except (OSError, ValueError):
        print('RunPod HTTP command failed; check configuration and pod ID.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
