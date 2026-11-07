"""Key, generation management and pod-side setup for the JAX compilation cache.

The weekly GPU benchmark runs on an ephemeral RunPod pod, so XLA's persistent
compilation cache only survives when it is written to storage the next pod can
see. A RunPod network volume provides that; this module owns everything about
it that is not the pod request itself:

* ``key``: the identity of one cache generation. It is a digest over the JAX
  version, the CUDA runtime/driver version and the sorted case matrix, so a
  re-pin of tenferro-rs keeps the cache (compiled programs depend on JAX, the
  CUDA toolchain and the shapes/ops, not on which tenferro revision produced
  the same shapes) while a JAX, CUDA, driver or case-matrix change starts a new
  generation.
* ``configure``: the pod-side entry point. It resolves the key, keeps the
  current generation plus the previous one, and exports the three JAX variables
  through ``$GITHUB_ENV``. With no network volume configured it prints one line
  and exports nothing, so the run continues with an empty cache.
* ``prune``: generation-level deletion only. Entries inside a generation are
  never touched individually; JAX owns what is inside its own directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import gpu_routine_selection  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'scripts/ci/runpod_config.json'
DEFAULT_MANIFEST = ROOT / 'benchmarks/gpu/manifests/routine.yaml'
# JAX's own default cap is unset (uncapped); the volume is shared, so bound it.
DEFAULT_MAX_SIZE_BYTES = 4 * 1024 ** 3
DEFAULT_MIN_COMPILE_SECS = 1
# Entries are per (case, backend) pair; text that cannot change a compiled
# program is dropped so editing a description does not invalidate a generation.
_NON_COMPILE_KEYS = frozenset({'run', 'verify', 'notes', 'description', 'intent', 'title'})
DISABLED_MESSAGE = 'jax compile cache: disabled (no network volume configured)'


def _selected_suites(manifest: Path, selection: str) -> set[str]:
    suites = yaml.safe_load(Path(manifest).read_text())['suites']
    known = {spec['file'] for spec in suites.values()}
    if not selection or selection.strip() == 'all':
        return known
    requested = {path.strip() for path in selection.split(',') if path.strip()}
    unknown = requested - known
    if unknown:
        raise ValueError(f'GPU suites absent from {Path(manifest)}: {sorted(unknown)}')
    return requested


def case_matrix(manifest: Path = DEFAULT_MANIFEST, selection: str = 'all',
                backends: str = '', vendor_arms: bool = False) -> list[str]:
    """One sorted entry per (case, JAX arm) the run will compile."""
    manifest = Path(manifest)
    document = yaml.safe_load(manifest.read_text())
    # Suite `file:` entries are repository-relative, so they always resolve
    # against the checkout, not against wherever the manifest happens to live.
    root = ROOT
    selected = _selected_suites(manifest, selection)
    filter_arms = {name.strip() for name in (backends or '').split(',') if name.strip()}
    entries: list[str] = []
    for suite_id, spec in document['suites'].items():
        if spec['file'] not in selected:
            continue
        source, problems, _instances = gpu_routine_selection.suite_problems(root, spec)
        arms = list(spec['backends'])
        if vendor_arms:
            arms += list(spec.get('vendor_backends') or [])
        jax_arms = sorted({arm for arm in arms
                           if arm.startswith('jax') and (not filter_arms or arm in filter_arms)})
        if not jax_arms:
            continue
        for problem_id in spec['problems']:
            case = {key: value for key, value in problems[problem_id].items()
                    if key not in _NON_COMPILE_KEYS}
            canonical = json.dumps(case, sort_keys=True, separators=(',', ':'))
            entries.extend(f'{suite_id}:{problem_id}:{arm}:{canonical}' for arm in jax_arms)
    return sorted(entries)


def cache_key(jax_version: str, cuda_version: str, driver_version: str,
              matrix: list[str]) -> str:
    """Digest of everything a compiled JAX executable depends on.

    The tenferro pin is deliberately absent: the same shapes/ops compile to the
    same executables whatever tenferro revision selected them, and including it
    would miss the cache every week.
    """
    if not jax_version:
        raise ValueError('a JAX version is required for the compilation-cache key')
    payload = '|'.join((jax_version, cuda_version or 'unknown', driver_version or 'unknown',
                        '\n'.join(matrix)))
    return hashlib.sha256(payload.encode()).hexdigest()


def _detect_jax_version() -> str:
    import jax
    return jax.__version__


def _detect_cuda_version() -> str:
    try:
        import torch
        return str(torch.version.cuda or '')
    except Exception:
        return ''


def _detect_driver_version() -> str:
    try:
        import subprocess
        output = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception:
        return ''
    versions = {line.strip() for line in output.splitlines() if line.strip()}
    # A mixed-driver set would make a single key meaningless; refuse rather
    # than pick one at random.
    if len(versions) == 1:
        return versions.pop()
    return ''


def configure_mount(config_path: Path = DEFAULT_CONFIG) -> str:
    try:
        config = json.loads(Path(config_path).read_text())
    except (OSError, ValueError):
        return '/opt/bench-cache'
    value = config.get('network_volume_mount_path') or '/opt/bench-cache'
    return str(value)


def prune_generations(root: Path, current: str, keep_previous: int = 1) -> list[str]:
    """Delete generations other than ``current`` and the newest ``keep_previous``.

    Deletion is per key directory. Nothing inside a kept generation is touched,
    because JAX owns that layout.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    others = []
    for entry in root.iterdir():
        if entry.name == current or not entry.is_dir():
            continue
        try:
            stamp = entry.stat().st_mtime
        except OSError:
            continue
        others.append((stamp, entry.name, entry))
    others.sort(key=lambda item: (item[0], item[1]), reverse=True)
    removed = []
    for _stamp, name, path in others[keep_previous:]:
        shutil.rmtree(path, ignore_errors=True)
        removed.append(name)
    return sorted(removed)


def _env_lines(cache_dir: Path, key: str, min_compile_secs: int, max_size_bytes: int) -> list[str]:
    return [f'JAX_COMPILATION_CACHE_DIR={cache_dir}',
            f'JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS={min_compile_secs}',
            f'JAX_PERSISTENT_CACHE_MAX_SIZE_BYTES={max_size_bytes}',
            f'BENCH_JAX_CACHE_KEY={key}']


def _emit(lines: list[str], github_env: Path | None) -> None:
    for line in lines:
        print(line)
    if github_env is not None:
        with Path(github_env).open('a', encoding='utf-8') as stream:
            stream.write('\n'.join(lines) + '\n')


def configure(volume_id: str, mount_path: str, selection: str = 'all', backends: str = '',
              jax_version: str = '', cuda_version: str = '', driver_version: str = '',
              min_compile_secs: int = DEFAULT_MIN_COMPILE_SECS,
              max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
              keep_previous: int = 1, manifest: Path = DEFAULT_MANIFEST,
              github_env: Path | None = None) -> dict:
    """Resolve the generation, prune older ones and export the JAX variables."""
    if not volume_id:
        print(DISABLED_MESSAGE)
        return {'enabled': False}
    jax_version = jax_version or _detect_jax_version()
    cuda_version = cuda_version or _detect_cuda_version()
    driver_version = driver_version or _detect_driver_version()
    matrix = case_matrix(manifest, selection, backends)
    key = cache_key(jax_version, cuda_version, driver_version, matrix)
    root = Path(mount_path) / 'jax'
    root.mkdir(parents=True, exist_ok=True)
    removed = prune_generations(root, key, keep_previous)
    cache_dir = root / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(cache_dir, os.W_OK):
        print(f'jax compile cache: disabled ({cache_dir} is not writable)')
        return {'enabled': False, 'key': key}
    _emit(_env_lines(cache_dir, key, min_compile_secs, max_size_bytes), github_env)
    print(f'jax compile cache: key={key} jax={jax_version} cuda={cuda_version or "unknown"} '
          f'driver={driver_version or "unknown"} cases={len(matrix)} removed={len(removed)}')
    return {'enabled': True, 'key': key, 'cache_dir': str(cache_dir),
            'cases': len(matrix), 'removed': removed}


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    key = commands.add_parser('key', help='print one compilation-cache key')
    key.add_argument('--jax-version', required=True)
    key.add_argument('--cuda-version', default='')
    key.add_argument('--driver-version', default='')
    key.add_argument('--suite', default='all')
    key.add_argument('--backends', default='')
    key.add_argument('--vendor-arms', action='store_true')
    key.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)

    configure_parser = commands.add_parser('configure', help='prepare the pod-side cache directory')
    configure_parser.add_argument('--volume-id', default='',
                                  help='RunPod network volume ID; empty disables the cache')
    configure_parser.add_argument('--mount-path', default='',
                                  help='volume mount path (default: runpod_config.json)')
    configure_parser.add_argument('--suite', default='all')
    configure_parser.add_argument('--backends', default='')
    configure_parser.add_argument('--jax-version', default='')
    configure_parser.add_argument('--cuda-version', default='')
    configure_parser.add_argument('--driver-version', default='')
    configure_parser.add_argument('--min-compile-secs', type=int, default=DEFAULT_MIN_COMPILE_SECS)
    configure_parser.add_argument('--max-size-bytes', type=int, default=DEFAULT_MAX_SIZE_BYTES)
    configure_parser.add_argument('--keep-previous', type=int, default=1)
    configure_parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    configure_parser.add_argument('--github-env', type=Path,
                                  help='append the exported variables here ($GITHUB_ENV)')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.command == 'key':
        matrix = case_matrix(args.manifest, args.suite, args.backends, args.vendor_arms)
        print(cache_key(args.jax_version, args.cuda_version, args.driver_version, matrix))
        return 0
    try:
        configure(args.volume_id, args.mount_path or configure_mount(),
                  selection=args.suite, backends=args.backends,
                  jax_version=args.jax_version, cuda_version=args.cuda_version,
                  driver_version=args.driver_version,
                  min_compile_secs=args.min_compile_secs,
                  max_size_bytes=args.max_size_bytes,
                  keep_previous=args.keep_previous,
                  manifest=args.manifest, github_env=args.github_env)
    except (OSError, ValueError, ImportError) as error:
        # An unusable volume is not a benchmark failure: collect cold instead of
        # losing the week's run to cache infrastructure.
        print(f'jax compile cache: disabled ({error})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
