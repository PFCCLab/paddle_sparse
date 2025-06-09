from __future__ import annotations

import paddle

from paddle_sparse.tensor import SparseTensor


def permute(src: SparseTensor, perm: paddle.Tensor) -> SparseTensor:
    assert src.is_quadratic()
    return src.index_select(0, perm).index_select(1, perm)


SparseTensor.permute = lambda self, perm: permute(self, perm)
