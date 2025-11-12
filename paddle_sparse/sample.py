from typing import Optional
from typing import Tuple

import paddle
import paddle_sparse_ops

from paddle_sparse.tensor import SparseTensor


def sample(
    src: SparseTensor, num_neighbors: int, subset: Optional[paddle.Tensor] = None
) -> paddle.Tensor:

    rowptr, col, _ = src.csr()
    rowcount = src.storage.rowcount()

    if subset is not None:
        rowcount = rowcount[subset]
        rowptr = rowptr[subset]
    else:
        rowptr = rowptr[:-1]

    rand = paddle.rand((rowcount.size(0), num_neighbors), device=col.place)
    rand.mul_(rowcount.to(rand.dtype).view(-1, 1))
    rand = rand.to(paddle.int64)
    rand.add_(rowptr.view(-1, 1))

    return col[rand]


def sample_adj(
    src: SparseTensor, subset: paddle.Tensor, num_neighbors: int, replace: bool = False
) -> Tuple[SparseTensor, paddle.Tensor]:

    rowptr, col, value = src.csr()

    rowptr, col, n_id, e_id = paddle_sparse_ops.sample_adj(
        rowptr, col, subset, num_neighbors, replace
    )

    if value is not None:
        value = value[e_id]

    out = SparseTensor(
        rowptr=rowptr,
        row=None,
        col=col,
        value=value,
        sparse_sizes=(subset.size(0), n_id.size(0)),
        is_sorted=True,
    )

    return out, n_id


SparseTensor.sample = sample
SparseTensor.sample_adj = sample_adj
