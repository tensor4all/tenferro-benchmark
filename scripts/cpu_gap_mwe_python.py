#!/usr/bin/env python3
"""PyTorch reference for examples/cpu_gap_mwe.rs; native layouts, identical values."""
import json
import sys
import time
import numpy as np
import torch


def fixture(operation):
    if operation in ("cast", "reshape"):
        n = 33554432
        raw = (np.arange(n, dtype=np.int64) % 17).astype(np.float64)
        x = torch.from_numpy(raw)
        if operation == "cast":
            def check(y):
                assert bool((y[:4096].double() == x[:4096]).all())
            return lambda: x.to(torch.float32), check, n*4
        # Same logical [8192,4096] values as Rust's column-major reshape.
        # Prepare the metadata-only view outside timing; clone materializes
        # a fresh output and preserves the column-major compact layout.
        view = x.reshape(4096,8192).T
        def check(y):
            assert tuple(y.shape) == (8192,4096)
            assert bool((y.T.flatten()[:4096] == x[:4096]).all())
        return lambda: view.clone(), check, n*8
    if operation == "ifft-pattern":
        # Exact periodic values used by the maintained FFT suite.
        i = np.arange(2048, dtype=np.uint64)
        def values(seed):
            with np.errstate(over="ignore"):
                raw = i*np.uint64(6364136223846793005) + np.uint64(seed)*np.uint64(1442695040888963407)
            return ((raw % 2048).astype(np.float32)-1024)/1024
        block = values(17)+1j*values(18)
        x = torch.from_numpy(np.tile(block, 512))
        expected = torch.from_numpy(np.fft.ifft(block.astype(np.complex128)))
        def check(y):
            assert float((y[::512]-expected).abs().max()) < 1e-5
        return lambda: torch.fft.ifft(x, norm="backward"), check, 1048576*8
    if operation == "ifft":
        n = 1048576
        x = torch.ones(n, dtype=torch.complex64)
        def check(y):
            assert abs(complex(y[0]) - 1) < 1e-5
            assert float(y[1:].abs().max()) < 1e-5
        return lambda: torch.fft.ifft(x, norm="backward"), check, n*8
    if operation == "diagonal":
        n = 8388608
        x = torch.zeros(n, 2, 2, dtype=torch.float64)
        x[:, 0, 0] = 1
        x[:, 1, 1] = 2
        def check(y):
            assert bool((y[:, 0] == 1).all()) and bool((y[:, 1] == 2).all())
        return lambda: torch.diagonal(x, dim1=1, dim2=2).clone(), check, n*16
    if operation in ("triangular", "lstsq"):
        m, n, nrhs = (4096, 4096, 64) if operation == "triangular" else (768, 384, 16)
        i = np.arange(m, dtype=np.int64)[:, None]
        j = np.arange(n, dtype=np.int64)[None, :]
        a = ((i*17+j*13) % 19 - 9).astype(np.float64)
        a /= m * (100 if operation == "triangular" else 10)
        if operation == "triangular":
            a = np.tril(a, -1) + np.eye(n)
        else:
            a[i == j] = 2
        rhs = np.repeat(a.sum(axis=1)[:, None], nrhs, axis=1)
        a, rhs = torch.from_numpy(a.copy()), torch.from_numpy(rhs.copy())
        def check(y):
            assert float((y-1).abs().max()) < 1e-10
        if operation == "triangular":
            op = lambda: torch.linalg.solve_triangular(a, rhs, upper=False)
        else:
            # gels returns more than solution; retain the full result, just as
            # the public allocation-returning operation requires.
            op = lambda: torch.linalg.lstsq(a, rhs, driver="gels")
            original_check = check
            check = lambda y: original_check(y.solution)
        return op, check, n*nrhs*8
    raise ValueError(operation)


def run(operation, threads, samples=15):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    op, check, retained = fixture(operation)
    check(op())
    for _ in range(3):
        op()
    cap = max(1, min(65536, (512 << 20)//retained))
    def batch(count):
        outputs = [None]*count
        started = time.perf_counter_ns()
        for i in range(count):
            outputs[i] = op()
        elapsed = time.perf_counter_ns()-started
        del outputs
        return elapsed
    count = 1
    while True:
        elapsed = batch(count)
        if elapsed >= 2_000_000 or count == cap:
            break
        count = min(count*2, cap)
    rows = [{"sample_index": i, "iterations": count, "elapsed_ns": batch(count)}
            for i in range(samples)]
    return {"samples": rows, "correctness_status": "passed",
            "calibration": {"iterations": count, "elapsed_ns": elapsed, "target_ns": 2000000},
            "provider": "mkl" if torch.backends.mkl.is_available() else "other",
            "torch_config": torch.__config__.show(), "threads_requested": threads,
            "scope": {"timer": [operation, "intrinsic output allocation", "Python dispatch"],
                      "outside_timer": ["fixture construction", "thread setup", "priming",
                                        "validation", "retention allocation", "destruction"]}}


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1], int(sys.argv[2]))))
