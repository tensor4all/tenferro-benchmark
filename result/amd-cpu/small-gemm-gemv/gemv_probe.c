// Diagnostic only: compare exact vector-shaped GEMM with the same provider's GEMV.
// No production dispatch change or size threshold. OpenBLAS CBLAS API, LP64.
#define _GNU_SOURCE
#include <cblas.h>
#include <dlfcn.h>
#include <stdlib.h>

typedef void (*gemm_fn)(enum CBLAS_ORDER, enum CBLAS_TRANSPOSE,
    enum CBLAS_TRANSPOSE, int, int, int, double, const double*, int,
    const double*, int, double, double*, int);
static gemm_fn original;
__attribute__((constructor)) static void setup(void) {
    original = (gemm_fn)dlsym(RTLD_NEXT, "cblas_dgemm");
    if (!original) abort();
}
void cblas_dgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE ta,
    enum CBLAS_TRANSPOSE tb, int m, int n, int k, double alpha,
    const double *a, int lda, const double *b, int ldb,
    double beta, double *c, int ldc) {
    if (order == CblasColMajor && m > 0 && n > 0 && k > 0 &&
        (ta == CblasNoTrans || ta == CblasTrans) &&
        (tb == CblasNoTrans || tb == CblasTrans)) {
        if (m == 1) {
            cblas_dgemv(order, tb == CblasNoTrans ? CblasTrans : CblasNoTrans,
                tb == CblasNoTrans ? k : n, tb == CblasNoTrans ? n : k,
                alpha, b, ldb, a, ta == CblasNoTrans ? lda : 1, beta, c, ldc);
            return;
        }
        if (n == 1) {
            cblas_dgemv(order, ta, ta == CblasNoTrans ? m : k,
                ta == CblasNoTrans ? k : m, alpha, a, lda,
                b, tb == CblasNoTrans ? 1 : ldb, beta, c, 1);
            return;
        }
    }
    original(order, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
