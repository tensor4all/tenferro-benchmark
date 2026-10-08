"""Identity and integrity checks for the fixed-prefix RunPod runtime bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tarfile
from pathlib import Path, PurePosixPath

PREFIX = '/opt/tenferro-benchmark-ci'
PYTHON_VERSION = '3.12.12'
UV_VERSION = '0.12.21'
CUDA_VERSION = '12.8'
INPUTS = ('pyproject.toml', 'uv.lock', 'scripts/ci/gpu_environment.py',
          'scripts/ci/prepare_gpu_environment.sh', 'scripts/ci/restore_gpu_environment.sh')


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def identity(backends: str, root: Path = Path('.')) -> dict:
    return {'schema': 1, 'platform': 'ubuntu-24.04-x86_64', 'prefix': PREFIX,
            'python': PYTHON_VERSION, 'uv': UV_VERSION, 'cuda': CUDA_VERSION,
            'jax_cuda': 'jax-cuda' in backends.split(','),
            'inputs': {name: digest(root / name) for name in INPUTS}}


def cache_key(value: dict) -> str:
    return 'runpod-runtime-v1-' + hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify(archive: Path, manifest: dict, expected: dict, expected_sha: str) -> None:
    if manifest['identity'] != expected:
        raise ValueError('GPU runtime bundle does not match the locked dependencies/platform')
    if manifest['archive_sha256'] != expected_sha or digest(archive) != expected_sha:
        raise ValueError('GPU runtime bundle digest mismatch')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('GPU runtime requires Linux x86_64')


def check_tar(stream) -> None:
    """Reject archive paths/links that could escape the fixed runtime prefix."""
    for member in stream:
        name = PurePosixPath(member.name)
        if name.is_absolute() or '..' in name.parts or name.parts[:1] != ('tenferro-benchmark-ci',):
            raise ValueError(f'Unexpected runtime archive path: {member.name}')
        if member.isdev() or member.isfifo():
            raise ValueError(f'Unexpected runtime archive entry: {member.name}')
        if member.issym() or member.islnk():
            target = PurePosixPath(member.linkname)
            if target.is_absolute():
                if not str(target).startswith(PREFIX + '/'):
                    raise ValueError(f'External runtime archive link: {member.linkname}')
            else:
                parts = list(name.parent.parts) if member.issym() else []
                for part in target.parts:
                    if part == '..':
                        if len(parts) <= 1:
                            raise ValueError(f'Escaping runtime archive link: {member.linkname}')
                        parts.pop()
                    elif part != '.':
                        parts.append(part)
                if parts[:1] != ['tenferro-benchmark-ci']:
                    raise ValueError(f'External runtime archive link: {member.linkname}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('key', 'manifest', 'verify', 'check-tar'))
    parser.add_argument('--backends', default='')
    parser.add_argument('--bundle', type=Path, default=Path('_gpu_environment'))
    parser.add_argument('--sha256')
    args = parser.parse_args()
    expected = identity(args.backends)
    archive = args.bundle / 'runtime.tar.zst'
    manifest = args.bundle / 'runtime.json'
    if args.action == 'key':
        print(cache_key(expected))
    elif args.action == 'manifest':
        manifest.write_text(json.dumps({'identity': expected,
                                       'archive_sha256': digest(archive)}, indent=2) + '\n')
    elif args.action == 'verify':
        verify(archive, json.loads(manifest.read_text()), expected, args.sha256)
        print('Locked GPU runtime bundle verified')
    else:
        import sys
        with tarfile.open(fileobj=sys.stdin.buffer, mode='r|') as stream:
            check_tar(stream)


if __name__ == '__main__':
    main()
