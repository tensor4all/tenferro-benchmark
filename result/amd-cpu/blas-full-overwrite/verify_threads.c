/* Diagnostic only: never LD_PRELOAD this into instruction/native measurements. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
void cblas_dgemm(int order,int ta,int tb,int m,int n,int k,double alpha,const double*a,int lda,const double*b,int ldb,double beta,double*c,int ldc) {
    typedef void (*gemm)(int,int,int,int,int,int,double,const double*,int,const double*,int,double,double*,int);
    gemm next=(gemm)dlsym(RTLD_NEXT,"cblas_dgemm");
    int (*threads)(void)=(int(*)(void))dlsym(RTLD_NEXT,"openblas_get_num_threads");
    if (!next || !threads || threads()!=1) abort();
    fprintf(stderr,"verified OpenBLAS threads=1 at dgemm %d %d %d\n",m,n,k);
    next(order,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}
