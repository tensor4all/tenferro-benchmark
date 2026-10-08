"""Prepared public oneMKL DFTI reference, with allocated Torch host outputs.

Only empty-output allocation and DftiComputeForward/Backward are executed by
call(). Descriptor creation/configuration/commit and thread setup are untimed.
Constants and signatures follow the installed public mkl_dfti.h (LP64 Linux).
"""
import ctypes as C
import os
from pathlib import Path


def prepared_fft(torch, x, operation, n, threads):
    library=Path(os.environ['MKLROOT'])/'lib/libmkl_rt.so'
    lib=C.CDLL(str(library))
    lib.DftiCreateDescriptor.argtypes=[C.POINTER(C.c_void_p),C.c_int,C.c_int,C.c_long]
    lib.DftiCreateDescriptor.restype=C.c_long
    lib.DftiSetValue.argtypes=[C.c_void_p,C.c_int];lib.DftiSetValue.restype=C.c_long
    lib.DftiCommitDescriptor.argtypes=[C.c_void_p];lib.DftiCommitDescriptor.restype=C.c_long
    lib.DftiFreeDescriptor.argtypes=[C.POINTER(C.c_void_p)];lib.DftiFreeDescriptor.restype=C.c_long
    lib.DftiErrorMessage.argtypes=[C.c_long];lib.DftiErrorMessage.restype=C.c_char_p
    for name in ('DftiComputeForward','DftiComputeBackward'):
        f=getattr(lib,name);f.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p];f.restype=C.c_long
    lib.mkl_set_num_threads_local.argtypes=[C.c_int];lib.mkl_set_num_threads_local.restype=C.c_int
    lib.mkl_get_max_threads.restype=C.c_int
    lib.mkl_set_num_threads_local(threads)
    def check(status):
        if status:raise RuntimeError(lib.DftiErrorMessage(status).decode())
    descriptor=C.c_void_p()
    double=x.dtype in (torch.float64,torch.complex128)
    real=operation in ('rfft','irfft')
    check(lib.DftiCreateDescriptor(C.byref(descriptor),C.c_int(36 if double else 35),C.c_int(33 if real else 32),C.c_long(1),C.c_long(n)))
    check(lib.DftiSetValue(descriptor,C.c_int(11),C.c_int(44)))  # out-of-place
    if real:check(lib.DftiSetValue(descriptor,C.c_int(10),C.c_int(39))) # complex-complex Hermitian storage
    if operation in ('ifft','irfft'):check(lib.DftiSetValue(descriptor,C.c_int(5),C.c_double(1/n)))
    check(lib.DftiCommitDescriptor(descriptor))
    output_dtype=(torch.float64 if double else torch.float32) if operation=='irfft' else (torch.complex128 if double else torch.complex64)
    output_shape=(n//2+1,) if operation=='rfft' else (n,)
    compute=lib.DftiComputeBackward if operation in ('ifft','irfft') else lib.DftiComputeForward
    source=C.c_void_p(x.data_ptr())
    def call():
        output=torch.empty(output_shape,dtype=output_dtype,device='cpu')
        check(compute(descriptor,source,C.c_void_p(output.data_ptr())))
        return output
    def cleanup():check(lib.DftiFreeDescriptor(C.byref(descriptor)))
    return call,cleanup,dict(library=str(library),provider='oneMKL DFTI',planning='create/configure/commit outside timer; first completed transform outside timer',max_threads=lib.mkl_get_max_threads())
