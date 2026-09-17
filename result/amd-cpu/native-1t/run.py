#!/usr/bin/env python3
"""Sequential existing-harness runs; monitor only, never change others' affinity."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time

root = Path(os.environ.get('OUT', Path(__file__).resolve().parent))
cpu = int(os.environ.get('CPU', '32'))
domain = range(cpu, cpu+8)
os.sched_setaffinity(0, {56})  # Monitor/client is outside the measured L3 domain.
container = 'tenferro-openblas-analysis'
container_id = subprocess.check_output(['docker','inspect',container,'--format','{{.Id}}'], text=True).strip()
stop = threading.Event()
samples = []

def cpu_stat():
    return {s.split()[0]: list(map(int, s.split()[1:9]))
            for s in Path('/proc/stat').read_text().splitlines()
            if s.startswith('cpu') and s.split()[0] != 'cpu'}

def monitor():
    while not stop.is_set():
        rows = subprocess.check_output(
            ['ps','-eLo','pid=,tid=,psr=,stat=,comm='], text=True).splitlines()
        foreign = []
        for row in rows:
            p = row.split()
            if len(p) >= 5 and int(p[2]) in domain and p[3].startswith('R'):
                try:
                    owned = container_id in Path(f'/proc/{p[0]}/cgroup').read_text()
                except FileNotFoundError:
                    owned = False
                if not owned:
                    foreign.append(row.strip())
        samples.append({'time':time.time(), 'cpu':cpu_stat(), 'foreign_runnable':foreign})
        stop.wait(0.5)

cases = ['bin_elementwise_mul_2048x2048','bin_matmul_1024','lm_batch_likelihood_sentence_3_12d']
variants = {
    'baseline': '/tmp/eager-views-candidate',
    'copy': '/tmp/canonical-copy-candidate',
    'gemv_probe': 'env LD_PRELOAD=/tmp/gemv_probe.so /tmp/canonical-copy-candidate',
    'pytorch': '.venv/bin/python scripts/benchmark_python.py --backend pytorch --num-threads 1',
}
thread = threading.Thread(target=monitor)
thread.start()
records = []
try:
    for case in cases:
        for variant, binary in variants.items():
            path = root / f'{case}-{variant}.log'
            if path.exists():
                raise RuntimeError(f'refuse to overwrite {path}')
            command = ('source scripts/thread_env.sh; configure_cpu_thread_env 1; '
                'export TENFERRO_MODE=eager TENFERRO_OPT_DOT_DECOMPOSER=0 '
                f'BENCH_INSTANCE={case} BENCH_RUNS=15 BENCH_WARMUPS=3; '
                f'timeout 90s taskset -c {cpu} {binary}')
            started = time.time()
            with path.open('w') as log:
                result = subprocess.run(['docker','exec',container,'bash','-c',command],
                    stdout=log, stderr=subprocess.STDOUT, timeout=100)
            records.append({'case':case,'variant':variant,'start':started,
                'end':time.time(),'exit_code':result.returncode,'command':command})
            print(case, variant, 'exit', result.returncode, flush=True)
            if result.returncode:
                raise RuntimeError('benchmark failed; inspect owned descendants before retry')
finally:
    stop.set()
    thread.join()
    (root/'activity.json').write_text(json.dumps(samples)+'\n')
    (root/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
