"""Matched CUDA PyTorch references for the dispatch/readback regression cases."""
import os
import time


def fixture(case, device):
    import torch
    p = case['params']
    kind = case['kind']

    def values(n, seed, dtype, *, decode=False):
        # Match Rust's f32 arithmetic for the decode fixture, f64 otherwise.
        i = torch.arange(n, dtype=torch.int64, device=device)
        v = ((i * 2654435761 + seed * 97) % 2001).to(dtype) / 1000 - 1
        return v if decode else v * 0.5

    if kind == 'prepared_decode':
        din, dout, length = p['in'], p['out'], p['len']
        # Rust: compact column-major x[in,len], W[in,out], result[len,out].
        x = values(din*length, 1, torch.float32, decode=True).reshape(length, din)
        w = values(din*dout, 2, torch.float32, decode=True).reshape(dout, din).T
        op = lambda: torch.matmul(x, w)
        expected = x.double() @ w.double()
        def check(out):
            torch.testing.assert_close(out.double(), expected, rtol=1e-3, atol=1e-3)
        return op, check, length*dout*4
    if kind == 'standard_batched_matmul':
        n, batch = p['n'], p['batch']
        # Keep native contiguous row-major operands with the same logical values.
        a = values(n*n*batch, 31, torch.float64).reshape(batch,n,n).transpose(1,2).contiguous()
        b = values(n*n*batch, 32, torch.float64).reshape(batch,n,n).transpose(1,2).contiguous()
        op = lambda: torch.bmm(a, b)
        # CPU scalar contractions are independent of the CUDA bmm route.
        ah, bh = a.cpu(), b.cpu()
        expected = torch.empty_like(ah)
        for batch_id in range(batch):
            for i in range(n):
                for j in range(n):
                    expected[batch_id,i,j] = sum(float(ah[batch_id,i,k])*float(bh[batch_id,k,j]) for k in range(n))
        def check(out):
            torch.testing.assert_close(out.cpu(), expected, rtol=1e-10, atol=1e-10)
        return op, check, n*n*batch*8
    if kind == 'linalg_readback_latency':
        n, operation = p['n'], p['op']
        a = values(n*n, 81, torch.float64).reshape(n,n).T.contiguous()
        a.diagonal().add_(n)
        b = values(n, 82, torch.float64).reshape(n,1)
        if operation == 'solve':
            op = lambda: torch.linalg.solve(a,b)
            def check(out):
                torch.testing.assert_close(a @ out, b, rtol=1e-10, atol=1e-10)
            return op, check, n*8
        if operation == 'qr':
            op = lambda: torch.linalg.qr(a)
            def check(out):
                q,r = out
                torch.testing.assert_close(q @ r, a, rtol=1e-10, atol=1e-10)
                torch.testing.assert_close(q.T @ q, torch.eye(n,device=device,dtype=a.dtype), rtol=1e-10, atol=1e-10)
            return op, check, 2*n*n*8
    raise ValueError(f'unsupported CUDA reference kind: {kind}')


def run(case, config, correctness_only=False, device=0):
    import torch
    torch.cuda.set_device(device)
    op, check, retained_bytes = fixture(case, f'cuda:{device}')
    first = op()  # Explicit lazy initialization even with zero warmups.
    check(first)
    torch.cuda.synchronize(device)
    del first
    row = dict(correctness_status='passed', samples=[], execution_path='pytorch',
               scope=dict(timer=['one public PyTorch CUDA operation', 'batch-edge device synchronization'],
                          outside_timer=['fixture, upload and layout conversion', 'lazy initialization',
                                         'validation and download', 'retention allocation and output destruction']))
    if correctness_only:
        return row
    ramp_ms = int(os.environ.get('GPU_BENCH_CLOCK_RAMP_MS', '300'))
    deadline = time.perf_counter() + ramp_ms/1000
    while time.perf_counter() < deadline:
        output = op()
        torch.cuda.synchronize(device)
        del output
    row['clock_ramp_ms'] = ramp_ms

    def batch(count):
        outputs = [None]*count
        start = time.perf_counter_ns()
        for i in range(count):
            outputs[i] = op()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter_ns()-start
        del outputs
        return elapsed

    for _ in range(config['warmups']):
        batch(1)
    target = int(config['min_runtime_ms']*1_000_000)
    cap = min(1 << 16, max(1, (1 << 30)//max(retained_bytes, 1)))
    iterations = 1
    deadline = time.monotonic() + 30
    while batch(iterations) < target and iterations < cap:
        if time.monotonic() > deadline:
            raise RuntimeError('GPU reference calibration deadline exceeded')
        iterations = min(iterations*2, cap)
    row['calibration'] = dict(target_ns=target, iterations=iterations)
    row['samples'] = [dict(sample_index=i, elapsed_ns=batch(iterations), iterations=iterations)
                      for i in range(config['runs'])]
    return row
