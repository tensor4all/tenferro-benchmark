#!/usr/bin/env python3
"""Check the image's PyTorch BLAS/LAPACK provider; no performance measurement."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, choices=(1, 4), default=1)
    args = parser.parse_args()
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    config = torch.__config__.show()
    assert re.search(r"BLAS_INFO=open(?:blas)?\b", config), config
    assert not torch.backends.mkl.is_available(), config
    assert torch.version.cuda is None, torch.__version__

    root = Path(os.environ["OPENBLAS_ROOT"])
    expected = (root / "lib/libopenblas.so").resolve(strict=True)
    torch_cpu = Path(torch.__file__).parent / "lib/libtorch_cpu.so"
    linked = subprocess.check_output(["ldd", str(torch_cpu)], text=True)
    libraries = re.findall(r"libopenblas\S*\s+=>\s+(\S+)", linked)
    assert libraries and all(Path(p).resolve() == expected for p in libraries), linked
    assert not re.search(r"libmkl\S*", linked), linked

    lib = ctypes.CDLL(str(expected))
    lib.openblas_get_config.restype = ctypes.c_char_p
    lib.openblas_get_parallel.restype = ctypes.c_int
    lib.openblas_get_num_threads.restype = ctypes.c_int
    blas_config = lib.openblas_get_config().decode()
    assert "USE64BITINT" not in blas_config, blas_config
    assert lib.openblas_get_parallel() == 1, "expected pthread OpenBLAS"
    # Ensure both BLAS and LAPACK exports exist in this same library.
    assert lib.dgemm_ and lib.dgesv_

    a = torch.tensor([[4., 1.], [1., 3.]], dtype=torch.float64)
    b = torch.tensor([1., 2.], dtype=torch.float64)
    torch.testing.assert_close(a @ a, torch.tensor([[17., 7.], [7., 10.]], dtype=torch.float64))
    torch.testing.assert_close(a @ torch.linalg.solve(a, b), b)
    torch.testing.assert_close(torch.linalg.eigh(a)[0].sum(), a.trace())
    assert torch.get_num_threads() == args.threads
    assert lib.openblas_get_num_threads() == args.threads
    print(json.dumps({
        "torch": torch.__version__, "torch_commit": torch.version.git_version,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "openblas_threads": lib.openblas_get_num_threads(),
        "openblas_library": str(expected), "openblas_config": blas_config,
        "torch_config": config, "linked_libraries": linked,
    }, indent=2))


if __name__ == "__main__":
    main()
