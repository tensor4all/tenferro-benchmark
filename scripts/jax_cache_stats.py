"""Cheap accounting for the JAX persistent compilation cache directory.

Timing rows must not infer a cache hit; the only evidence a row carries is the
directory's own growth (`cache_bytes_delta`, `cache_entries_delta`) recorded
around the case. JAX writes one flat file per compiled executable, so a
single-level scan is enough and stays cheap enough to run per case.
"""
from __future__ import annotations

import os


def stats(directory) -> dict:
    """Entries, regular files and bytes directly under ``directory``."""
    if directory is None:
        return {'entries': 0, 'inodes': 0, 'bytes': 0, 'available': False}
    entries = inodes = total = 0
    try:
        with os.scandir(os.fspath(directory)) as listing:
            for entry in listing:
                entries += 1
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    inodes += 1
                    total += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
    except OSError:
        return {'entries': 0, 'inodes': 0, 'bytes': 0, 'available': False}
    return {'entries': entries, 'inodes': inodes, 'bytes': total, 'available': True}


def delta(before: dict, after: dict) -> dict:
    """The growth between two :func:`stats` snapshots."""
    if not before.get('available') or not after.get('available'):
        return {'cache_bytes_delta': None, 'cache_entries_delta': None}
    return {'cache_bytes_delta': after['bytes'] - before['bytes'],
            'cache_entries_delta': after['entries'] - before['entries']}
