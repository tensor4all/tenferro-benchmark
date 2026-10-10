import argparse,hashlib,json,os,shutil,stat,subprocess,time,zipfile
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
root=parser.parse_args().root
protocol=json.loads((root/'protocol.json').read_text())
os.sched_setaffinity(0,set(protocol['settings']['cpu_affinity']))
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def snapshot():
 rows={}
 for line in Path('/proc/stat').read_text().splitlines():
  k,*v=line.split()
  if k in {f'cpu{i}' for i in protocol['settings']['cpu_affinity']}:
   v=list(map(int,v));rows[k]=(sum(v[:8]),v[3]+v[4])
 return rows
def idle():
 a=snapshot();time.sleep(.5);b=snapshot()
 return {k:1-(b[k][1]-v[1])/max(1,b[k][0]-v[0]) for k,v in a.items()}
def timed(label,call):
 before=idle();load=os.getloadavg();t=time.monotonic();call();elapsed=time.monotonic()-t;after=idle()
 row={'label':label,'seconds':elapsed,'load_start':load,'cpu_busy_before':before,'cpu_busy_after':after}
 result['timings'].append(row);save();print(json.dumps(row),flush=True)
def save(): (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
def manifest(path):
 out={}
 for p in sorted(path.rglob('*')):
  if p.is_symlink():out[str(p.relative_to(path))]={'link':os.readlink(p)}
  elif p.is_file():out[str(p.relative_to(path))]={'size':p.stat().st_size,'sha256':digest(p)}
 return out
def unpack_wheels(payload):
 target=payload/'wheels-unpacked';target.mkdir()
 for wheel in sorted((payload/'wheels').glob('*.whl')):
  with zipfile.ZipFile(wheel) as z:z.extractall(target/wheel.stem)
 for p in target.glob('*/nvidia/cuda_nvcc/bin/*'):p.chmod(p.stat().st_mode|stat.S_IXUSR|stat.S_IXGRP|stat.S_IXOTH)
 return target
result={'protocol':protocol,'timings':[],'cases':[]}
archive=root/'runtime.tar.zst'
assert not archive.exists(), 'Screen output exists; inspect it instead of restarting'
with archive.open('wb') as out:
 for n in range(5):
  part=root/'common'/f'{n:02}'/f'runtime.part{n:02}'
  assert part.stat().st_size>0
  with part.open('rb') as src:shutil.copyfileobj(src,out)
expected=(root/'common/00/runtime.sha256').read_text().split()[0]
assert digest(archive)==expected,'Original artifact checksum mismatch'
result['original']={'bytes':archive.stat().st_size,'sha256':expected};save()
baseline=root/'baseline';baseline.mkdir()
def baseline_prepare():
 subprocess.run(['tar','--zstd','-xf',str(archive),'-C',str(baseline)],check=True)
 # The production step first copies the wheels before extracting them.
 copied=root/'baseline-copied-wheels';shutil.copytree(baseline/'wheels',copied)
 target=baseline/'wheels-unpacked';target.mkdir()
 for w in sorted(copied.glob('*.whl')):
  with zipfile.ZipFile(w) as z:z.extractall(target/w.stem)
 for p in target.glob('*/nvidia/cuda_nvcc/bin/*'):p.chmod(p.stat().st_mode|0o111)
timed('baseline extract common, copy and unpack every wheel',baseline_prepare)
base_manifest=manifest(baseline)
(root/'baseline-files.json').write_text(json.dumps(base_manifest,indent=2)+'\n')
# Copy only the complete extracted payload; omit the redundant ZIP containers.
prepared=root/'prepared'
shutil.copytree(baseline,prepared,symlinks=True,ignore=shutil.ignore_patterns('wheels'))
expected_manifest={k:v for k,v in base_manifest.items() if not k.startswith('wheels/')}
assert manifest(prepared)==expected_manifest
for level in [3,10]:
 output=root/f'prepared-{level}.tar.zst'
 timed(f'compress prepared common zstd {level}',lambda:subprocess.run(['tar','-I',f'zstd -T4 -{level}','-cf',str(output),'-C',str(prepared),'.'],check=True,timeout=300))
 decoded=root/f'decoded-{level}';decoded.mkdir()
 timed(f'extract prepared common zstd {level}',lambda:subprocess.run(['tar','--zstd','-xf',str(output),'-C',str(decoded)],check=True,timeout=300))
 same=manifest(decoded)==expected_manifest
 assert same,'Prepared payload file content mismatch'
 row={'level':level,'bytes':output.stat().st_size,'sha256':digest(output),'all_files_identical':same,'file_count':len(expected_manifest),'byte_reduction':1-output.stat().st_size/archive.stat().st_size}
 result['cases'].append(row);save();print(json.dumps(row),flush=True)
result['status']='complete';save()
