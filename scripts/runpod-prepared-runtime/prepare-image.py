"""Assemble an image from the checksum-verified, frozen CI runtime bundles."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifacts', type=Path, required=True)
parser.add_argument('--context', type=Path, required=True)
args = parser.parse_args()
run = '38034449617'
args.context.mkdir(parents=True, exist_ok=False)
runtime = args.context / 'payload/opt/ci-cost-runtime'
runtime.mkdir(parents=True)
manifest = {'baseline_commit': '356a2011ef0f9d2d345ed31201fbe838995329a2',
            'source_run': int(run), 'bundles': {}, 'prepared_pjrt': True,
            'cuda_tiers': ['12.6', '12.8'], 'cutensor': '2.6.0.4',
            'jax_cuda12_pjrt': '0.10.2', 'cudnn': '9.23.2.1', 'nvcc': '12.9.86'}
for kind, stem, destination in [('common', 'runtime', runtime),
                                ('cuda12.6', 'sdk', runtime / 'cuda-12.6'),
                                ('cuda12.8', 'sdk', runtime / 'cuda-12.8')]:
    prefix = f'gpu-runtime-{run}-{kind}-part'
    archive = args.context / f'{kind}.tar.zst'
    with archive.open('wb') as output:
        for index in range(5):
            part = args.artifacts / f'{prefix}{index:02}' / f'{stem}.part{index:02}'
            if not part.is_file() or part.stat().st_size == 0:
                raise RuntimeError(f'Missing runtime part: {part}')
            with part.open('rb') as source:
                shutil.copyfileobj(source, output)
    expected = (args.artifacts / f'{prefix}00' / f'{stem}.sha256').read_text().split()[0]
    with archive.open('rb') as source:
        actual = hashlib.file_digest(source, 'sha256').hexdigest()
    if actual != expected:
        raise RuntimeError(f'Checksum mismatch in {kind}')
    manifest['bundles'][kind] = {'sha256': actual, 'bytes': archive.stat().st_size}
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(['tar', '--zstd', '-xf', str(archive), '-C', str(destination)], check=True)
    archive.unlink()

wheels = sorted((runtime / 'wheels').glob('*.whl'))
if len(wheels) != 3:
    raise RuntimeError('Expected exactly three pinned PJRT wheels')
for wheel in wheels:
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(runtime / 'wheels-unpacked' / wheel.stem)
shutil.rmtree(runtime / 'wheels')
for executable in (runtime / 'wheels-unpacked').glob('*/nvidia/cuda_nvcc/bin/*'):
    executable.chmod(executable.stat().st_mode | 0o111)
# Move the existing cuTENSOR tree into its execution location, avoiding a second copy.
shutil.move(str(runtime / 'opt/tenferro-ci'), str(args.context / 'payload/opt/tenferro-ci'))
for tier in manifest['cuda_tiers']:
    sdk = runtime / f'cuda-{tier}'
    if not (sdk / '.seed-complete').is_file():
        raise RuntimeError(f'Incomplete CUDA SDK {tier}')
    library_dir = sdk / 'targets/x86_64-linux/lib'
    for name in ['nvrtc', 'cublas', 'cusolver', 'cusparse']:
        link = library_dir / f'lib{name}.so'
        if not link.exists():
            matches = sorted(library_dir.glob(f'lib{name}.so.*'))
            if not matches:
                raise RuntimeError(f'Missing {name} in {tier}')
            link.symlink_to(matches[0].name)
    cuda_link = args.context / f'payload/usr/local/cuda-{tier}'
    cuda_link.parent.mkdir(parents=True, exist_ok=True)
    cuda_link.symlink_to(f'/opt/ci-cost-runtime/cuda-{tier}')
(runtime / 'prepared-image.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
