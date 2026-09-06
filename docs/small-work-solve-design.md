# Small-work concrete solve slice

Use the existing `TensorLinalgExt::solve` API and canonical
`linalg.solve.ordinary.concrete` contract. Six F64 column-major cases cover n2/4/16
with fresh and shared backend sessions. Each workflow makes one solve call;
session entry/exit is inside fresh timing and outside shared timing. Inputs and
numerical checking stay outside both; solve and output lifetime stay inside.
No new library API, backend adapter or dependency is needed.

A is non-diagonal, nonsymmetric and strictly row-diagonally dominant. Construct
an independent known, nonuniform X and B=A X using the existing host triple-loop
matrix-product oracle. Check the full computed solution against X and check its
residual with that independent product. This is not a diagonal or identity-system
shortcut. Check that solving leaves the borrowed inputs unchanged.

Do not label concrete calls as eager. The current inventory has no eager solve
contract. This bounded slice does not claim solve AD, compiled, other dtypes,
noncontiguous/changing metadata or performance acceptance. Those remain separate
coverage/evidence requirements; it adds a representative linalg family without
expanding an unrestricted Cartesian product.
