"""Run with the diagnostic DSO preloaded; all preparation/checks are untimed."""
import ctypes as c
import math

lib = c.CDLL(None)
lib.openblas_get_num_threads.restype = c.c_int
assert lib.openblas_get_num_threads() == 1
f = lib.cblas_dgemm
f.argtypes = [c.c_int]*6 + [c.c_double, c.POINTER(c.c_double), c.c_int,
    c.POINTER(c.c_double), c.c_int, c.c_double, c.POINTER(c.c_double), c.c_int]
count = 0
for m,n,k in [(1,144,11),(1,11,11),(1,1,11),(144,1,11),(11,1,11),
              (12,12,12),(144,12,12),(12,12,144)]:
    for ta in [111,112]:
        for tb in [111,112]:
            ar,ac = (m,k) if ta == 111 else (k,m)
            br,bc = (k,n) if tb == 111 else (n,k)
            lda,ldb,ldc = ar+2,br+2,m+3
            a = (c.c_double*(lda*ac))(*[(i%19-9)/16 for i in range(lda*ac)])
            b = (c.c_double*(ldb*bc))(*[(i%13-6)/8 for i in range(ldb*bc)])
            before_a,before_b = bytes(a),bytes(b)
            for alpha,beta in [(1.,0.),(.75,-.25),(0.,0.)]:
                out = (c.c_double*(ldc*n))(*([777.]*(ldc*n)))
                expected = list(out)
                for j in range(n):
                    for i in range(m):
                        s = math.fsum(a[i+q*lda if ta==111 else q+i*lda]*
                            b[q+j*ldb if tb==111 else j+q*ldb] for q in range(k))
                        out[i+j*ldc] = float('nan') if beta == 0 else .5
                        expected[i+j*ldc] = alpha*s + beta*.5
                f(102,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,out,ldc)
                assert all(math.isclose(x,y,abs_tol=1e-12,rel_tol=1e-12)
                           for x,y in zip(out,expected)), (m,n,k,ta,tb,alpha,beta)
                assert bytes(a)==before_a and bytes(b)==before_b
                count += 1
print(f'{count} nonzero/padded/transpose/accumulation cases passed; OpenBLAS threads=1')
