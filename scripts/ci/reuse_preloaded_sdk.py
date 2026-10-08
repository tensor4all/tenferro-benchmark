#!/usr/bin/env python3
"""Reuse only byte-identical SDK libraries across the published-image boundary."""
import argparse
import hashlib
import json
from pathlib import Path


def reuse(sdk: Path, image_sdk: Path, manifest: dict, *, prepare: bool) -> int:
    transferred = sdk / 'targets/x86_64-linux/lib'
    preloaded = image_sdk / 'targets/x86_64-linux/lib'
    saved = 0
    for name, entry in manifest.items():
        source = transferred / name if prepare else preloaded / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f'Required library is not a regular file: {source}')
        if source.stat().st_size != entry['bytes']:
            raise ValueError(f'Preloaded library size differs: {name}')
        digest = hashlib.sha256()
        with source.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != entry['sha256']:
            raise ValueError(f'Preloaded library content differs: {name}')
        saved += entry['bytes']
    # Validate all entries before changing the staged SDK.
    for name in manifest:
        target = transferred / name
        if prepare:
            target.unlink()
        else:
            target.symlink_to(preloaded / name)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--image-sdk', type=Path, default=Path('/usr/local/cuda-12.8'))
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('ci_preloaded_sdk_manifest.json').read_text())
    print(f'Byte-identical preloaded SDK bytes reused: {reuse(args.sdk, args.image_sdk, manifest, prepare=args.prepare)}')

if __name__ == '__main__':
    main()
