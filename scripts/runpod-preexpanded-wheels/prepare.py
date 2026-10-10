"""Validate the actual producer's wheel transform against every frozen file."""
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
parser.add_argument('--producer', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(exist_ok=False)
archive = args.output / 'original.tar.zst'
with archive.open('wb') as output:
    for index in range(5):
        part = args.artifacts / f'runtime.part{index:02}'
        assert part.stat().st_size > 0
        with part.open('rb') as source:
            shutil.copyfileobj(source, output)
def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()
expected_sha = (args.artifacts / 'runtime.sha256').read_text().split()[0]
assert sha(archive) == expected_sha
payload = args.output / 'payload'
payload.mkdir()
subprocess.run(['tar', '--zstd', '-xf', str(archive), '-C', str(payload)], check=True)
def inventory(root):
    result = {}
    for path in root.rglob('*'):
        name = str(path.relative_to(root))
        if path.is_symlink(): result[name] = {'link': os.readlink(path)}
        elif path.is_file(): result[name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return result
expected = {k:v for k,v in inventory(payload).items() if not k.startswith('wheels/')}
wheels = sorted((payload / 'wheels').glob('*.whl'))
assert len(wheels) == 3
for wheel in wheels:
    with zipfile.ZipFile(wheel) as zipped:
        for member in zipped.infolist():
            if member.is_dir(): continue
            with zipped.open(member) as source:
                expected[f'wheels-unpacked/{wheel.stem}/{member.filename}'] = {
                    'sha256': hashlib.file_digest(source, 'sha256').hexdigest(), 'bytes': member.file_size}
script = args.producer.read_text()
# Execute the exact changed transform, independent of network installers and
# unchanged CUDA SDK assembly. Unit tests exercise the full packaging helper.
start = script.index('  python3 - "$payload" <<\'PY\'\n')
end = script.index('\nfi\n', start)
subprocess.run(['bash', '-euc', 'payload="$1"\n' + script[start:end], 'prepare', str(payload)], check=True)
assert inventory(payload) == expected
prepared = args.output / 'runtime.tar.zst'
subprocess.run(['tar', '--zstd', '-cf', str(prepared), '-C', str(payload), '.'], check=True)
reduction = 1 - prepared.stat().st_size / archive.stat().st_size
assert reduction >= .05, 'Predeclared byte-reduction gate failed'
decoded = args.output / 'decoded'
decoded.mkdir()
subprocess.run(['tar', '--zstd', '-xf', str(prepared), '-C', str(decoded)], check=True)
assert inventory(decoded) == expected
for ptxas in decoded.glob('wheels-unpacked/*/nvidia/cuda_nvcc/bin/ptxas'):
    assert os.access(ptxas, os.X_OK)
    subprocess.run([str(ptxas), '--version'], check=True)
plugin = next(decoded.glob('wheels-unpacked/*/jax_plugins/xla_cuda12/xla_cuda_plugin.so'))
assert 'GetPjrtApi' in subprocess.check_output(['nm', '-D', str(plugin)], text=True)
record = {'source_run': 38034449617, 'original_bytes': archive.stat().st_size,
          'prepared_bytes': prepared.stat().st_size, 'byte_reduction': reduction,
          'original_sha256': expected_sha, 'prepared_sha256': sha(prepared),
          'all_files_identical': True, 'file_count': len(expected),
          'compression': 'unchanged tar --zstd default level 3',
          'scope': 'transfer bytes and content preservation; no elapsed-time claim',
          'files': expected}
(args.output / 'verification.json').write_text(json.dumps(record, indent=2)+'\n')
(args.output / 'runtime.sha256').write_text(record['prepared_sha256']+'  runtime.tar.zst\n')
subprocess.run(['split', '-n', '5', '-d', '-a', '2', 'runtime.tar.zst', 'runtime.part'], cwd=args.output, check=True)
for index in range(5):
    dest = args.output / f'{index:02}'
    dest.mkdir()
    shutil.move(args.output / f'runtime.part{index:02}', dest)
shutil.copy(args.output / 'runtime.sha256', args.output / '00/runtime.sha256')
print(json.dumps({k:v for k,v in record.items() if k != 'files'}))
