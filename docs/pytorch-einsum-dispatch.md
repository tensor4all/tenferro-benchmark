# PyTorch Binary Einsum Dispatch Notes

This note records how the benchmark's PyTorch baseline dispatches binary
einsum cases, with emphasis on the non-compact batched outer-product diagnostic.

Source snapshots used for this investigation:

- PyTorch source checkout: `/Users/hiroshi/projects/tensor4all/pytorch`,
  commit `52b7da3f54b`
- Benchmark Python environment: PyTorch `2.12.0`, opt_einsum `3.4.0`
- Benchmark runner: `scripts/benchmark_python.py`

## Benchmark Entry Point

The PyTorch baseline in this repository does not call `torch.einsum` directly.
It calls opt_einsum with a precomputed path:

```python
oe.contract(fmt, *operands, optimize=path, backend="torch")
```

For binary benchmark instances, `path` is always `[(0, 1)]`. This means the
runner still enters opt_einsum, but there is only one contraction step.

## Relevant Cases

| Instance | Row-major equation | Input shapes | Expected output | Observed output stride |
|---|---|---|---|---|
| `bin_batched_outer_product_compact_j16_k16_o64_t64` | `tkj,to->tokj` | `(64,16,16)`, `(64,64)` | `(64,64,16,16)` | `(16384,256,16,1)`, contiguous |
| `bin_batched_outer_product_noncompact_j16_k16_o64_t64` | `tjk,to->tokj` | `(64,16,16)`, `(64,64)` | `(64,64,16,16)` | `(16384,256,1,16)`, non-contiguous |
| `bin_outer_product_4096` | `i,j->ij` | `(4096)`, `(4096)` | `(4096,4096)` | contiguous |

The compact and non-compact batched cases perform the same scalar operation.
The difference is output layout: the non-compact case asks for `j,k` in an order
that produces a non-contiguous output stride.

## Dispatch Tree

### 1. `torch.einsum` frontend

In PyTorch's `torch/functional.py`, direct `torch.einsum` bypasses PyTorch's
own opt_einsum path when the number of operands is two or fewer:

```text
if len(operands) <= 2 or not opt_einsum.enabled:
    return _VF.einsum(equation, operands)
```

This is not the exact entry point used by the benchmark, but it explains an
important baseline: binary `torch.einsum(...)` goes directly to ATen
`einsum`.

### 2. opt_einsum runner path

The benchmark goes through opt_einsum. opt_einsum computes a contraction list
and chooses between backend `tensordot` and backend `einsum`.

For the two batched outer-product cases, the contraction list is:

```text
noncompact: [((1, 0), frozenset(), 'to,tjk->tokj', ('tokj',), False)]
compact:    [((1, 0), frozenset(), 'to,tkj->tokj', ('tokj',), False)]
```

The final `False` is opt_einsum's BLAS flag. Therefore opt_einsum calls
backend `einsum`, not backend `tensordot`.

For the plain vector outer product:

```text
plain_outer: [((1, 0), frozenset(), 'j,i->ij', ('ij',), 'OUTER/EINSUM')]
```

This is still executed via backend `einsum` because the BLAS flag contains
`EINSUM`.

### 3. ATen native `einsum`

ATen implements `einsum` in `aten/src/ATen/native/Linear.cpp`.

The high-level flow is:

1. Parse input and output labels.
2. Align all operands to `output_dims + contraction_dims` by inserting
   singleton dimensions and permuting metadata.
3. Contract operands pairwise.
4. For each pair, call `sumproduct_pair(a, b, sum_dims, true)`.

For outer-product cases there are no contraction dimensions, so `sum_dims` is
empty. `sumproduct_pair` exits immediately through:

```text
if (sum_dims_.empty())
    return at::mul(left_, right_);
```

So the CPU route for these benchmark cases is:

```text
opt_einsum.contract
  -> torch.einsum
    -> aten::einsum
      -> unsqueeze/permute metadata alignment
      -> aten::mul(left, right)
```

It does not call `aten::bmm`, `aten::outer`, `torch.tensordot`, or BLAS.

### 4. `bmm_outer_product` is not used here

The PyTorch source tree contains a native override named
`torch/_native/ops/bmm_outer_product`. It only registers an override for
`aten::bmm` on CUDA and XPU dispatch keys. Its predicate also requires 3D input
shapes like `(B, M, 1)` and `(B, 1, N)`.

The benchmarked CPU outer-product cases do not reach that override:

- they are CPU runs,
- they do not call `aten::bmm`, because `sum_dims` is empty,
- they are handled by broadcasted `aten::mul`.

## Profiler Evidence

The following profiler-filtered op list was observed for both compact and
non-compact batched outer products:

```text
aten::einsum
aten::unsqueeze
aten::as_strided
aten::permute
aten::mul
```

No `aten::bmm`, `aten::outer`, `aten::mm`, `aten::matmul`, `aten::clone`,
`aten::copy_`, or `aten::contiguous` event appears in the filtered profile for
these cases.

Observed output layouts:

```text
noncompact tjk,to->tokj:
  shape  = (64, 64, 16, 16)
  stride = (16384, 256, 1, 16)
  contiguous = False

compact tkj,to->tokj:
  shape  = (64, 64, 16, 16)
  stride = (16384, 256, 16, 1)
  contiguous = True
```

Both cases reach the same broad ATen operation class: broadcasted multiply. The
main semantic difference is whether the resulting TensorIterator output layout
is contiguous.

## ATen `mul` Implementation

The `aten::mul.Tensor` schema is a structured TensorIterator op:

```text
mul.Tensor(Tensor self, Tensor other) -> Tensor
  structured_delegate: mul.out

mul.out(Tensor self, Tensor other, *, Tensor out) -> Tensor
  structured_inherits: TensorIteratorBase
  CPU, CUDA, MPS, MTIA: mul_out
```

The dense CPU path is:

```text
aten::mul.Tensor
  -> mul.out structured op
    -> TensorIteratorBase::build_borrowing_binary_op(...)
    -> TORCH_IMPL_FUNC(mul_out)
      -> mul_stub(device_type(), iterator)
        -> cpu::mul_kernel(iterator)
          -> cpu_kernel_vec(iterator, scalar_lambda, vector_lambda)
```

For `float64` tensor-tensor multiplication, `cpu::mul_kernel` takes the normal
dispatch branch:

```text
scalar:     return a * b
vectorized: return Vectorized<double>(a) * Vectorized<double>(b)
```

The special branches are not used for this benchmark:

- `bool` uses logical `&&`.
- `complex half` uses an opmath temporary.
- reduced floating scalar multiplication has a scalar-specialized branch.

### TensorIterator behavior

TensorIterator is the key optimization layer here. For binary `mul`, it:

1. infers the broadcasted output shape,
2. assigns stride `0` to broadcasted input dimensions,
3. reorders dimensions by stride to improve inner-loop locality,
4. allocates the output with a compatible non-overlapping dense stride,
5. coalesces adjacent iteration dimensions when legal,
6. calls the CPU loop.

With one CPU thread, `TensorIteratorBase::for_each` falls through to
`serial_for_each`. The pointwise kernel itself is still vectorized through
`cpu_kernel_vec` when the inner dimension is contiguous, allowing one input to
be a stride-0 broadcast scalar in that inner loop.

For the non-compact batched outer-product case, the aligned `mul` inputs are:

```text
a = y.unsqueeze(2).unsqueeze(3)
  shape  = (64, 64, 1, 1)
  stride = (64, 1, 1, 1)

b = x.unsqueeze(1).permute(0, 1, 3, 2)
  shape  = (64, 1, 16, 16)
  stride = (256, 256, 1, 16)
```

Then `a * b` directly produces:

```text
shape  = (64, 64, 16, 16)
stride = (16384, 256, 1, 16)
contiguous = False
```

That is the same output layout observed from `torch.einsum("tjk,to->tokj", ...)`.
So the non-compact layout is not a post-einsum view; it is the stride chosen for
the allocated `aten::mul` result.

For the compact case, the aligned second operand is:

```text
b = x.unsqueeze(1)
  shape  = (64, 1, 16, 16)
  stride = (256, 256, 16, 1)
```

Then `a * b` produces the contiguous output stride:

```text
stride = (16384, 256, 16, 1)
```

This explains why PyTorch handles both compact and non-compact batched outer
products well: it does not need a dedicated outer-product kernel. The generic
TensorIterator `mul` path can allocate the output in the requested dense
layout, preserve broadcasted stride-0 inputs, reorder iteration dimensions, and
use a vectorized inner loop.

## Implications for tenferro Investigation

The current PyTorch result does not indicate a hidden CPU outer-product kernel
or BLAS path. For the non-compact batched outer-product diagnostic, PyTorch is
fast because the operation is a broadcasted multiply with metadata-only input
alignment and a non-contiguous output stride where needed.

Therefore the tenferro trace gap for
`bin_batched_outer_product_noncompact_j16_k16_o64_t64` should be investigated
as a tenferro trace/executor routing issue, not as a missing PyTorch-like BLAS
outer-product call.

Next checks should be:

1. Compare tenferro trace lowered ops for compact vs non-compact batched
   outer product.
2. Verify whether the traced path enters a generic einsum extension path while
   eager enters the specialized batched outer-product route.
3. Check whether trace materializes a permutation or workspace that eager and
   PyTorch avoid.
4. If trace has to produce a non-contiguous output, verify whether it can return
   the same kind of lazy owned view as other traced TensorValue outputs.

The separate PyTorch gap in `bin_batched_matmul_b32_m128_n128_k128` should be
handled independently. That case has contraction dimensions and is expected to
flow through `sumproduct_pair`'s `permute -> reshape -> bmm -> view -> permute`
pipeline, unlike outer-product cases.
