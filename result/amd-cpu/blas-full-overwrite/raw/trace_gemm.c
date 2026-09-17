/* Diagnostic-only LP64 OpenBLAS call tracing, not for timed execution. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
void cblas_dgemm(int order,int ta,int tb,int m,int n,int k,double alpha,const double*a,int lda,const double*b,int ldb,double beta,double*c,int ldc) {
    typedef void (*fn)(int,int,int,int,int,int,double,const double*,int,const double*,int,double,double*,int);
    fn next=(fn)dlsym(RTLD_NEXT,"cblas_dgemm");
    if(!next) abort();
    fprintf(stderr,"GEMM %d %d %d %d %d %d %d %d %d\n",m,n,k,order,ta,tb,lda,ldb,ldc);
    next(order,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}
