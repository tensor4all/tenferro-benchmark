"""Render a self-contained watchdog before handing the startup script to RunPod."""
import argparse
import json
import shlex
from pathlib import Path


def render(api_url: str, lifetime: int) -> str:
    if lifetime <= 0:
        raise ValueError('watchdog lifetime must be positive')
    source = Path(__file__).with_name('runpod_http.py').read_text()
    return f'''set -euo pipefail
cat > /tmp/runpod_http.py <<'RUNPOD_HTTP_SOURCE'
{source}
RUNPOD_HTTP_SOURCE
# The watchdog is independent of the Actions runner and reads only pod-scoped credentials.
# Remove any inherited POD_ID so this pod can only delete RUNPOD_POD_ID.
unset POD_ID
printf '%s\\n' {shlex.quote(api_url)} > /tmp/pod-watchdog-api-url
: "${{RUNPOD_POD_ID:?missing pod ID}}" "${{RUNPOD_API_KEY:?missing pod-scoped API key}}"
echo 'Pod-scoped credentials present; Python watchdog installed.' > /tmp/pod-watchdog.status
nohup python3 /tmp/runpod_http.py delete --api-url {shlex.quote(api_url)} --delay {lifetime} >/tmp/pod-watchdog.log 2>&1 &
echo $! > /tmp/pod-watchdog.pid
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--lifetime', type=int, required=True)
    args = parser.parse_args()
    print(render(json.loads(args.config.read_text())['api_url'], args.lifetime), end='')


if __name__ == '__main__':
    main()
