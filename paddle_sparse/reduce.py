from typing import Optional

import paddle
from paddle_scatter import scatter
from paddle_scatter import segment_csr

from paddle_sparse.tensor import SparseTensor


def reduction(
    src: SparseTensor, dim: Optional[int] = None, reduce: str = "sum"
) -> paddle.Tensor:
    value = src.storage.value()

    if dim is None:
        if value is not None:
            if reduce == "sum" or reduce == "add":
                return value.sum()
            elif reduce == "mean":
                return value.mean()
            elif reduce == "min":
                return value.min()
            elif reduce == "max":
                return value.max()
            else:
                raise ValueError
        else:
            if reduce == "sum" or reduce == "add":
                return paddle.full([], src.nnz(), dtype=src.dtype()).to(src.device())
            elif reduce == "mean" or reduce == "min" or reduce == "max":
                return paddle.full([], 1, dtype=src.dtype()).to(src.device())
            else:
                raise ValueError
    else:
        if dim < 0:
            dim = src.dim() + dim

        if dim == 0 and value is not None:
            col = src.storage.col()
            return scatter(value, col, 0, None, src.size(1), reduce)
        elif dim == 0 and value is None:
            if reduce == "sum" or reduce == "add":
                return src.storage.colcount().to(src.dtype())
            elif reduce == "mean" or reduce == "min" or reduce == "max":
                return paddle.ones(src.size(1), dtype=src.dtype())
            else:
                raise ValueError
        elif dim == 1 and value is not None:
            return segment_csr(value, src.storage.rowptr(), None, reduce)
        elif dim == 1 and value is None:
            if reduce == "sum" or reduce == "add":
                return src.storage.rowcount().to(src.dtype())
            elif reduce == "mean" or reduce == "min" or reduce == "max":
                return paddle.ones(src.size(0), dtype=src.dtype())
            else:
                raise ValueError
        elif dim > 1 and value is not None:
            if reduce == "sum" or reduce == "add":
                return value.sum(axis=dim - 1)
            elif reduce == "mean":
                return value.mean(axis=dim - 1)
            elif reduce == "min":
                return value.min(axis=dim - 1)[0]
            elif reduce == "max":
                return value.max(axis=dim - 1)[0]
            else:
                raise ValueError
        else:
            raise ValueError


def sum(src: SparseTensor, dim: Optional[int] = None) -> paddle.Tensor:
    return reduction(src, dim, reduce="sum")


def mean(src: SparseTensor, dim: Optional[int] = None) -> paddle.Tensor:
    return reduction(src, dim, reduce="mean")


def min(src: SparseTensor, dim: Optional[int] = None) -> paddle.Tensor:
    return reduction(src, dim, reduce="min")


def max(src: SparseTensor, dim: Optional[int] = None) -> paddle.Tensor:
    return reduction(src, dim, reduce="max")


SparseTensor.sum = lambda self, dim=None: sum(self, dim)
SparseTensor.mean = lambda self, dim=None: mean(self, dim)
SparseTensor.min = lambda self, dim=None: min(self, dim)
SparseTensor.max = lambda self, dim=None: max(self, dim)
