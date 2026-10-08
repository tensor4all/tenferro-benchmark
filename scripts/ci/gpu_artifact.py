"""Bind a hosted GPU benchmark build to its sources and binary digest."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def identity(binary: Path) -> dict[str, str]:
    def commit(directory: str) -> str:
        return subprocess.check_output(
            ['git', '-C', directory, 'rev-parse', 'HEAD'], text=True
        ).strip()
    return {
        'harness_commit': commit('.'),
        'tenferro_commit': commit('extern/tenferro-rs'),
        'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'features': 'cuda',
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('binary', type=Path)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    manifest = args.binary.with_suffix('.json')
    expected = identity(args.binary)
    if args.write:
        manifest.write_text(json.dumps(expected, indent=2) + '\n')
    elif json.loads(manifest.read_text()) != expected:
        raise SystemExit('GPU benchmark artifact does not match sources or binary digest')
    else:
        print('Fresh hosted GPU benchmark artifact verified')


if __name__ == '__main__':
    main()
