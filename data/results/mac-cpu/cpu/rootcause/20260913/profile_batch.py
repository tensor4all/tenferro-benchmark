import os,subprocess,time,pathlib
root=pathlib.Path('/tmp/tenferro-rootcause'); env=os.environ.copy();env.update(RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',VECLIB_NUM_THREADS='1',TENFERRO_CPU_BACKEND_KIND='blas',PUBLICATION_GATE_PROFILE='full',PUBLICATION_GATE_SUITE='batched',PUBLICATION_GATE_TENFERRO_MODE='both',CPU_OPS_BENCHMARK_FILTER='batched_solve,grad_sum_batched_solve_backward',BENCH_WARMUPS='3',BENCH_RUNS='60')
with (root/'batch_profile_run.csv').open('w') as f:
 p=subprocess.Popen(['target/release/publication_gate'],env=env,stdout=f)
 time.sleep(.3)
 subprocess.run(['sample',str(p.pid),'5','1','-file',str(root/'batch_sample.txt')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 assert p.wait()==0
