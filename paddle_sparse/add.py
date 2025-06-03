from typing import Optional

import paddle
from paddle import Tensor
from paddle_scatter import gather_csr

from paddle_sparse.tensor import SparseTensor


def add(src, other):  # noqa: F811
    if isinstance(other, Tensor):
        rowptr, col, value = src.csr()
        if other.shape[0] == src.size(0) and other.shape[1] == 1:  # Row-wise.
            other = gather_csr(other.squeeze(1), rowptr)
        elif other.shape[0] == 1 and other.shape[1] == src.size(1):  # Col-wise.
            other = other.squeeze(0)[col]
        else:
            raise ValueError(
                f"Size mismatch: Expected size ({src.size(0)}, 1, ...) or "
                f"(1, {src.size(1)}, ...), but got size {other.size()}."
            )
        if value is not None:
            value = other.astype(value.dtype).add_(value)
        else:
            value = other.add_(paddle.full([], 1, dtype=other.dtype))
        return src.set_value(value, layout="coo")

    elif isinstance(other, SparseTensor):
        rowA, colA, valueA = src.coo()
        rowB, colB, valueB = other.coo()

        row = paddle.concat([rowA, rowB], axis=0)
        col = paddle.concat([colA, colB], axis=0)

        value: Optional[Tensor] = None
        if valueA is not None and valueB is not None:
            value = paddle.concat([valueA, valueB], axis=0)

        M = max(src.size(0), other.size(0))
        N = max(src.size(1), other.size(1))
        sparse_sizes = (M, N)

        out = SparseTensor(row=row, col=col, value=value, sparse_sizes=sparse_sizes)
        out = out.coalesce(reduce="sum")
        return out

    else:
        raise NotImplementedError


def add_(src: SparseTensor, other: paddle.Tensor) -> SparseTensor:
    rowptr, col, value = src.csr()
    if other.size(0) == src.size(0) and other.size(1) == 1:  # Row-wise.
        other = gather_csr(other.squeeze(1), rowptr)
    elif other.size(0) == 1 and other.size(1) == src.size(1):  # Col-wise.
        other = other.squeeze(0)[col]
    else:
        raise ValueError(
            f"Size mismatch: Expected size ({src.size(0)}, 1, ...) or "
            f"(1, {src.size(1)}, ...), but got size {other.shape}."
        )

    if value is not None:
        value = value.add_(other.astype(value.dtype))
    else:
        value = other.add_(paddle.full([], 1, dtype=other.dtype))
    return src.set_value_(value, layout="coo")


def add_nnz(
    src: SparseTensor, other: paddle.Tensor, layout: Optional[str] = None
) -> SparseTensor:
    value = src.storage.value()
    if value is not None:
        value = value.add(other.astype(value.dtype))
    else:
        value = other.add(paddle.full([], 1, dtype=other.dtype))
    return src.set_value(value, layout=layout)


def add_nnz_(
    src: SparseTensor, other: paddle.Tensor, layout: Optional[str] = None
) -> SparseTensor:
    value = src.storage.value()
    if value is not None:
        value = value.add_(other.astype(value.dtype))
    else:
        value = other.add(paddle.full([], 1, dtype=other.dtype))
    return src.set_value_(value, layout=layout)


SparseTensor.add = lambda self, other: add(self, other)
SparseTensor.add_ = lambda self, other: add_(self, other)
SparseTensor.add_nnz = lambda self, other, layout=None: add_nnz(self, other, layout)
SparseTensor.add_nnz_ = lambda self, other, layout=None: add_nnz_(self, other, layout)
SparseTensor.__add__ = SparseTensor.add
SparseTensor.__radd__ = SparseTensor.add
SparseTensor.__iadd__ = SparseTensor.add_
