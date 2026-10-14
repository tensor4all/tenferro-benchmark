"""Single-operation PyTorch references for #177; all preparation is untimed."""


def torch_case(case):
    import torch
    from benchmark_perf_issues import _values

    p, kind = case["params"], case["kind"]
    n = p["n"]
    if kind == "umbrella_add":
        a = torch.tensor(_values(n, 110), dtype=torch.float64)
        b = torch.tensor(_values(n, 111), dtype=torch.float64)
        want = torch.tensor([x+y for x,y in zip(_values(n,110), _values(n,111))], dtype=torch.float64)
        if p["route"] == "into_held":
            out = torch.empty_like(a)
            op = lambda: torch.add(a,b,out=out)
            retained = 0
            scope = ["torch.add out= caller output; no session concept"]
        else:
            op = lambda: torch.add(a,b)
            retained = n*8
            scope = ["torch.add allocating; no session concept"]
    elif kind == "umbrella_matmul":
        av, bv = _values(n*n,112), _values(n*n,113)
        a = torch.tensor(av,dtype=torch.float64).reshape(n,n).T
        b = torch.tensor(bv,dtype=torch.float64).reshape(n,n).T
        want = torch.tensor([[sum(av[i+n*k]*bv[k+n*j] for k in range(n))
                              for j in range(n)] for i in range(n)],dtype=torch.float64)
        op = lambda: torch.matmul(a,b)
        retained, scope = n*n*8, ["torch.matmul allocating"]
    elif kind == "umbrella_neg":
        vals = _values(n,115)
        a = torch.tensor(vals,dtype=torch.float64)
        want = torch.tensor([-v for v in vals],dtype=torch.float64)
        op = lambda: torch.neg(a)
        retained, scope = n*8, ["torch.neg allocating"]
    elif kind == "umbrella_gemm":
        a = torch.ones((n,n),dtype=torch.float64)
        b = torch.full((n,n),2.,dtype=torch.float64)
        want = torch.full((n,n),2.*n,dtype=torch.float64)
        op = lambda: torch.matmul(a,b)
        retained, scope = n*n*8, ["torch.matmul allocating; equivalent prepared binary contraction"]
    elif kind == "umbrella_lu" and p["op"] == "solve":
        batch = p["batch"]
        a = torch.tensor(_values(n*n*batch,116),dtype=torch.float64).reshape(batch,n,n).transpose(-1,-2)
        a.diagonal(dim1=-2,dim2=-1).add_(n)
        a[:,[0,1],:] = a[:,[1,0],:]
        b = a.sum(-1,keepdim=True)
        lu, pivots = torch.linalg.lu_factor(a)
        want = torch.ones_like(b)
        op = lambda: torch.linalg.lu_solve(lu,pivots,b)
        retained, scope = n*batch*8, ["torch.linalg.lu_solve only; factorization prepared outside timing"]
    else:
        raise ValueError(f"no matching single PyTorch operation for {kind}")

    def check():
        got = op()
        if got.shape != want.shape or not torch.isfinite(got).all():
            return float("inf")
        return float(((got-want).abs()/want.abs().clamp(min=1.)).max())
    return op, check, retained, scope
