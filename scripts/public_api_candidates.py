"""Explicit setup and numerical validation for the public-API gap cases."""
from __future__ import annotations


class PreparedOperation:
    def __init__(self, setup, operations_per_sample=1):
        self.setup = setup
        self.operation = None
        self.validator = None
        self.operations_per_sample = operations_per_sample

    def prepare(self):
        if self.operation is None:
            self.operation, self.validator = self.setup()
            self.validate()  # lazy initialization, compilation and checks are untimed

    def validate(self):
        self.validator(self.operation())

    def __call__(self):
        if self.operation is None:
            raise RuntimeError("operation must be prepared before sampling")
        return self.operation()


def qr_fixture(rows, cols, deficient=False):
    """Identical logical values to candidates.rs, in native NumPy layout."""
    import numpy as np

    indices = np.arange(rows * cols, dtype=np.int64)
    values = ((indices * 1837 + 335) % 2048 - 1024) / 1024.0
    a = values.reshape((rows, cols), order="F").copy()
    a *= 0.125 / rows
    a[np.arange(cols), np.arange(cols)] += 2.0 + np.arange(cols) / cols
    if deficient:
        a[:, 3 * cols // 4:] = 0.0
    return a


def pytorch_linalg_cases(shapes=((256, 128), (768, 384)), solve_sizes=(256, 768)):
    import torch
    rows = []
    for m, n in shapes:
        def setup(m=m, n=n):
            a = torch.from_numpy(qr_fixture(m, n))
            def validate(output):
                packed, tau = output
                q = torch.linalg.householder_product(packed, tau)
                r = torch.triu(packed[:n, :])
                torch.testing.assert_close(q @ r, a, rtol=1e-10, atol=1e-10)
                torch.testing.assert_close(q.T @ q, torch.eye(n, dtype=a.dtype), rtol=1e-10, atol=1e-10)
            return lambda: torch.geqrf(a), validate
        rows.append(("cpu/linalg_uncovered", "householder_qr", "f64", f"{m}x{n}",
                     "compact QR; setup=explicit; output retained; validation=untimed", PreparedOperation(setup, 16)))
        for name in ["rank_revealing_qr", "rank_revealing_qr_deficient"]:
            rows.append(("cpu/linalg_uncovered", name, "f64", f"{m}x{n}",
                         "no direct public column-pivoted QR returning Q/R/permutation/numerical-rank", None))
    for n in solve_sizes:
        def setup(n=n):
            a = torch.from_numpy(qr_fixture(n, n))
            indices = torch.arange(n * 16, dtype=torch.int64)
            b = (((indices * 1837 + 2*335) % 2048 - 1024).to(torch.float64) / 1024).reshape(16,n).T.contiguous()
            out = torch.full_like(b, float("nan"))
            def validate(output):
                if output.data_ptr() != out.data_ptr():
                    raise AssertionError("solve must reuse caller output")
                torch.testing.assert_close(a @ out, b, rtol=1e-10, atol=1e-10)
            return lambda: torch.linalg.solve(a,b,out=out), validate
        rows.append(("cpu/output_reuse", "solve_read_into", "f64", f"{n}x{n},rhs=16",
                     "solve includes factorization; destination=preallocated; setup=explicit; validation=untimed", PreparedOperation(setup, 16)))
    return rows


def jax_linalg_cases(shapes=((256, 128), (768, 384)), solve_sizes=(256, 768)):
    import jax
    import jax.numpy as jnp
    import numpy as np
    rows=[]
    for m,n in shapes:
        def setup(m=m,n=n):
            a=jnp.asarray(qr_fixture(m,n))
            op=jax.jit(lambda x: jnp.linalg.qr(x,mode="raw"))
            def validate(output):
                h,tau=map(np.asarray,jax.block_until_ready(output))
                # NumPy/JAX raw QR stores transposed reflectors. Apply the
                # reflector sequence only during validation, outside timing.
                packed=h.T
                q=np.eye(m,n)
                for j in reversed(range(n)):
                    v=np.zeros(m); v[j]=1; v[j+1:]=packed[j+1:,j]
                    q-=tau[j]*np.outer(v,v @ q)
                r=np.triu(packed[:n,:])
                np.testing.assert_allclose(q @ r,np.asarray(a),rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(q.T @ q,np.eye(n),rtol=1e-10,atol=1e-10)
            return lambda: op(a),validate
        rows.append(("cpu/linalg_uncovered","householder_qr","f64",f"{m}x{n}",
                     "compact raw QR; setup/XLA compilation=explicit untimed; validation=untimed",PreparedOperation(setup, 16)))
        for name in ["rank_revealing_qr","rank_revealing_qr_deficient"]:
            rows.append(("cpu/linalg_uncovered",name,"f64",f"{m}x{n}",
                         "jax.scipy.linalg.qr(pivoting=True) returns Q/R/permutation but no numerical rank; four-output contract has no direct equivalent",None))
    for n in solve_sizes:
        rows.append(("cpu/output_reuse","solve_read_into","f64",f"{n}x{n},rhs=16",
                     "JAX solve has no mutable caller-owned out API",None))
    return rows


def selection_operation(framework, condition, on_true, on_false):
    def setup():
        import numpy as np
        c=condition.get(); x=on_true.get(); y=on_false.get()
        if framework == "pytorch":
            import torch
            expected=torch.from_numpy(np.where(c.numpy(),x.numpy(),y.numpy()))
            def validate(out): torch.testing.assert_close(out,expected,rtol=0,atol=0)
            return lambda: torch.where(c,x,y),validate
        import jax
        import jax.numpy as jnp
        expected=np.where(np.asarray(c),np.asarray(x),np.asarray(y))
        op=jax.jit(jnp.where)
        def validate(out): np.testing.assert_array_equal(np.asarray(jax.block_until_ready(out)),expected)
        return lambda:op(c,x,y),validate
    return PreparedOperation(setup)


def masked_log_softmax_operation(value,mask):
    def setup():
        import jax
        import numpy as np
        x=value.get(); m=mask.get()
        source=np.asarray(x); include=np.asarray(m)
        masked=np.where(include,source,-np.inf)
        shifted=masked-np.max(masked,axis=1,keepdims=True)
        expected=shifted-np.log(np.sum(np.exp(shifted),axis=1,keepdims=True))
        op=jax.jit(lambda x,m:jax.nn.log_softmax(x,axis=1,where=m))
        def validate(out): np.testing.assert_allclose(np.asarray(jax.block_until_ready(out)),expected,rtol=1e-10,atol=1e-10)
        return lambda:op(x,m),validate
    return PreparedOperation(setup, operations_per_sample=16)
