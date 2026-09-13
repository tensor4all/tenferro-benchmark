import os,subprocess,pathlib
root=pathlib.Path('/tmp/tenferro-rootcause')
# Diagnostic design: same small f64 cases for every configuration; three fresh
# processes/configuration; 10 warmups, 101 samples. Change Rayon and native BLAS
# independently. Do not promote these timings into publication reports.
for rep in range(3):
 for rayon,blas in [(1,1),(4,1),(1,4),(4,4)]:
  env=os.environ.copy()
  for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','BLIS_NUM_THREADS']:
   env[name]=str(blas)
  env.update(RAYON_NUM_THREADS=str(rayon),TENFERRO_CPU_BACKEND_KIND='blas',PUBLICATION_GATE_PROFILE='full',PUBLICATION_GATE_SUITE='small',PUBLICATION_GATE_TENFERRO_MODE='both',CPU_OPS_BENCHMARK_FILTER='matmul,grad_sum_qr_jvp,grad_sum_qr_vjp',BENCH_WARMUPS='10',BENCH_RUNS='101')
  out=root/f'threads_rep{rep}_r{rayon}_b{blas}.csv'
  with out.open('w') as f: subprocess.run(['target/release/publication_gate'],env=env,stdout=f,check=True)
  print(out.name,flush=True)
