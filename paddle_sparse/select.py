from __future__ import annotations

from paddle_sparse.narrow import narrow
from paddle_sparse.tensor import SparseTensor


def select(src: SparseTensor, dim: int, idx: int) -> SparseTensor:
    return narrow(src, dim, start=idx, length=1)


SparseTensor.select = lambda self, dim, idx: select(self, dim, idx)
