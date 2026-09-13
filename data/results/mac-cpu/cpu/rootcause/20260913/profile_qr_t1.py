import os,subprocess,time,pathlib
root=pathlib.Path('/tmp/tenferro-rootcause'); env=os.environ.copy();env.update(RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',VECLIB_NUM_THREADS='1',TENFERRO_CPU_BACKEND_KIND='blas',PUBLICATION_GATE_PROFILE='quick',PUBLICATION_GATE_SUITE='small',PUBLICATION_GATE_TENFERRO_MODE='trace',CPU_OPS_BENCHMARK_FILTER='grad_sum_qr_vjp',BENCH_WARMUPS='10',BENCH_RUNS='20000')
with (root/'qr_profile_run_t1.csv').open('w') as f:
 p=subprocess.Popen(['target/release/publication_gate'],env=env,stdout=f)
 time.sleep(.5)
 subprocess.run(['sample',str(p.pid),'3','1','-file',str(root/'qr_sample_t1.txt')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 assert p.wait()==0
