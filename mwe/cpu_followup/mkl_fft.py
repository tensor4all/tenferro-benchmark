"""Prepared public oneMKL DFTI reference, with allocated Torch host outputs.

Only empty-output allocation and DftiComputeForward/Backward are executed by
call(). Descriptor creation/configuration/commit and thread setup are untimed.
Constants and signatures follow the installed public mkl_dfti.h (LP64 Linux).
"""
import ctypes as C
import ctypes.util
import os
from pathlib import Path


class MklRuntimeUnavailable(RuntimeError):
    """No loadable oneMKL reference runtime (not a DFTI execution failure)."""


def load_runtime():
    """Search at runtime only; never set MKLROOT in the build environment."""
    candidates = []
    root = os.environ.get('MKLROOT')
    if root:
        candidates.extend(str(Path(root) / suffix) for suffix in
                          ('lib/libmkl_rt.so', 'lib/intel64/libmkl_rt.so'))
    system = ctypes.util.find_library('mkl_rt')
    if system:
        candidates.append(system)
    candidates.extend(('/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so',
                       '/usr/lib/x86_64-linux-gnu/libmkl_rt.so'))
    errors = []
    for library in dict.fromkeys(candidates):
        try:
            return C.CDLL(library), library
        except OSError as error:
            errors.append(f'{library}: {error}')
    raise MklRuntimeUnavailable('oneMKL runtime not found: install oneMKL or set MKLROOT for the reference process; '
                                + '; '.join(errors))


def prepared_fft(torch, x, operation, n, threads):
    lib,library=load_runtime()
    lib.DftiCreateDescriptor.argtypes=[C.POINTER(C.c_void_p),C.c_int,C.c_int,C.c_long]
    lib.DftiCreateDescriptor.restype=C.c_long
    lib.DftiSetValue.argtypes=[C.c_void_p,C.c_int];lib.DftiSetValue.restype=C.c_long
    lib.DftiCommitDescriptor.argtypes=[C.c_void_p];lib.DftiCommitDescriptor.restype=C.c_long
    lib.DftiFreeDescriptor.argtypes=[C.POINTER(C.c_void_p)];lib.DftiFreeDescriptor.restype=C.c_long
    lib.DftiErrorMessage.argtypes=[C.c_long];lib.DftiErrorMessage.restype=C.c_char_p
    for name in ('DftiComputeForward','DftiComputeBackward'):
        f=getattr(lib,name);f.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p];f.restype=C.c_long
    lib.MKL_Set_Num_Threads_Local.argtypes=[C.c_int];lib.MKL_Set_Num_Threads_Local.restype=C.c_int
    lib.MKL_Get_Max_Threads.restype=C.c_int
    lib.MKL_Set_Num_Threads_Local(threads)
    def check(status):
        if status:raise RuntimeError(lib.DftiErrorMessage(status).decode())
    descriptor=C.c_void_p()
    double=x.dtype in (torch.float64,torch.complex128)
    real=operation in ('rfft','irfft')
    check(lib.DftiCreateDescriptor(C.byref(descriptor),C.c_int(36 if double else 35),C.c_int(33 if real else 32),C.c_long(1),C.c_long(n)))
    check(lib.DftiSetValue(descriptor,C.c_int(11),C.c_int(44)))  # out-of-place
    if real:check(lib.DftiSetValue(descriptor,C.c_int(10),C.c_int(39))) # complex-complex Hermitian storage
    if operation in ('ifft','irfft'):check(lib.DftiSetValue(descriptor,C.c_int(5),C.c_double(1/n)))
    check(lib.DftiSetValue(descriptor,C.c_int(28),C.c_int(52))) # preserve input
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
    return call,cleanup,dict(library=str(library),provider='oneMKL DFTI',planning='create/configure/commit outside timer; first completed transform outside timer',max_threads=lib.MKL_Get_Max_Threads())
