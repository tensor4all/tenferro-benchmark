import os,subprocess,time,pathlib
root=pathlib.Path('/tmp/tenferro-rootcause'); env=os.environ.copy();env.update(RAYON_NUM_THREADS='4',OMP_NUM_THREADS='4',VECLIB_MAXIMUM_THREADS='4',VECLIB_NUM_THREADS='4',TENFERRO_CPU_BACKEND_KIND='blas',PUBLICATION_GATE_PROFILE='quick',PUBLICATION_GATE_SUITE='small',PUBLICATION_GATE_TENFERRO_MODE='trace',CPU_OPS_BENCHMARK_FILTER='grad_sum_qr_vjp',BENCH_WARMUPS='10',BENCH_RUNS='10000')
with (root/'qr_profile_run.csv').open('w') as f:
 p=subprocess.Popen(['target/release/publication_gate'],env=env,stdout=f)
 time.sleep(.5)
 subprocess.run(['sample',str(p.pid),'3','1','-file',str(root/'qr_sample.txt')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 assert p.wait()==0
