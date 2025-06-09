from __future__ import annotations

from typing import Optional

import paddle
from paddle import Tensor
from paddle_scatter import gather_csr

from paddle_sparse.tensor import SparseTensor


def mul(src, other):  # noqa: F811
    if isinstance(other, Tensor):
        rowptr, col, value = src.csr()
        if other.shape[0] == src.size(0) and other.shape[1] == 1:  # Row-wise...
            other = gather_csr(other.squeeze(1), rowptr)
            pass
        # Col-wise...
        elif other.shape[0] == 1 and other.shape[1] == src.size(1):
            other = other.squeeze(0)[col]
        else:
            raise ValueError(
                f"Size mismatch: Expected size ({src.size(0)}, 1, ...) or "
                f"(1, {src.size(1)}, ...), but got size {other.shape}."
            )

        if value is not None:
            value = other.astype(value.dtype).multiply_(
                paddle.full([], value, dtype=other.dtype)
            )
        else:
            value = other
        return src.set_value(value, layout="coo")

    assert isinstance(other, SparseTensor)

    if not src.is_coalesced():
        raise ValueError("The `src` tensor is not coalesced")
    if not other.is_coalesced():
        raise ValueError("The `other` tensor is not coalesced")

    rowA, colA, valueA = src.coo()
    rowB, colB, valueB = other.coo()

    row = paddle.concat([rowA, rowB], axis=0)
    col = paddle.concat([colA, colB], axis=0)

    if valueA is not None and valueB is not None:
        value = paddle.concat([valueA, valueB], axis=0)
    else:
        raise ValueError("Both sparse tensors must contain values")

    M = max(src.size(0), other.size(0))
    N = max(src.size(1), other.size(1))
    sparse_sizes = (M, N)

    # Sort indices:
    idx = paddle.full((col.numel() + 1,), -1, dtype=col.dtype)
    idx[1:] = row * sparse_sizes[1] + col
    perm = idx[1:].argsort()
    idx[1:] = idx[1:][perm]

    row, col, value = row[perm], col[perm], value[perm]

    valid_mask = idx[1:] == idx[:-1]
    valid_idx = valid_mask.nonzero().view([-1])

    return SparseTensor(
        row=row[valid_mask],
        col=col[valid_mask],
        value=value[valid_idx - 1] * value[valid_idx],
        sparse_sizes=sparse_sizes,
    )


def mul_(src: SparseTensor, other: paddle.Tensor) -> SparseTensor:
    rowptr, col, value = src.csr()
    if other.shape[0] == src.size(0) and other.shape[1] == 1:  # Row-wise...
        other = gather_csr(other.squeeze(1), rowptr)
        pass
    elif other.shape[0] == 1 and other.shape[1] == src.size(1):  # Col-wise...
        other = other.squeeze(0)[col]
    else:
        raise ValueError(
            f"Size mismatch: Expected size ({src.size(0)}, 1, ...) or "
            f"(1, {src.size(1)}, ...), but got size {other.shape}."
        )

    if value is not None:
        value = value.multiply_(other.astype(value.dtype))
    else:
        value = other
    return src.set_value_(value, layout="coo")


def mul_nnz(
    src: SparseTensor,
    other: paddle.Tensor,
    layout: Optional[str] = None,
) -> SparseTensor:
    value = src.storage.value()
    if value is not None:
        value = value.multiply_(other.astype(value.dtype))
    else:
        value = other
    return src.set_value(value, layout=layout)


def mul_nnz_(
    src: SparseTensor,
    other: paddle.Tensor,
    layout: Optional[str] = None,
) -> SparseTensor:
    value = src.storage.value()
    if value is not None:
        value = value.multiply_(other.astype(value.dtype))
    else:
        value = other
    return src.set_value_(value, layout=layout)


SparseTensor.mul = lambda self, other: mul(self, other)
SparseTensor.mul_ = lambda self, other: mul_(self, other)
SparseTensor.mul_nnz = lambda self, other, layout=None: mul_nnz(self, other, layout)
SparseTensor.mul_nnz_ = lambda self, other, layout=None: mul_nnz_(self, other, layout)
SparseTensor.__mul__ = SparseTensor.mul
SparseTensor.__rmul__ = SparseTensor.mul
SparseTensor.__imul__ = SparseTensor.mul_
