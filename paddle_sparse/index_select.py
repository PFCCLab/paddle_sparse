from __future__ import annotations

from typing import Optional

import paddle
from paddle_scatter import gather_csr

from paddle_sparse.storage import SparseStorage
from paddle_sparse.storage import get_layout
from paddle_sparse.tensor import SparseTensor


def index_select(src: SparseTensor, dim: int, idx: paddle.Tensor) -> SparseTensor:
    dim = src.dim() + dim if dim < 0 else dim
    assert idx.dim() == 1

    if dim == 0:
        old_rowptr, col, value = src.csr()
        rowcount = src.storage.rowcount()

        rowcount = rowcount[idx]

        rowptr = paddle.zeros([idx.shape[0] + 1], dtype=col.dtype)
        rowptr[1:] = paddle.cumsum(rowcount, axis=0)

        row = paddle.arange(idx.shape[0]).to(col.place).repeat_interleave(rowcount)

        perm = paddle.arange(row.shape[0]).to(row.place)
        perm += gather_csr(old_rowptr[idx] - rowptr[:-1], rowptr)

        col = col[perm]

        if value is not None:
            value = value[perm]

        sparse_sizes = (idx.shape[0], src.sparse_size(1))

        storage = SparseStorage(
            row=row,
            rowptr=rowptr,
            col=col,
            value=value,
            sparse_sizes=sparse_sizes,
            rowcount=rowcount,
            colptr=None,
            colcount=None,
            csr2csc=None,
            csc2csr=None,
            is_sorted=True,
        )
        return src.from_storage(storage)

    elif dim == 1:
        old_colptr, row, value = src.csc()
        colcount = src.storage.colcount()

        colcount = colcount[idx]

        colptr = paddle.zeros([idx.shape[0] + 1], dtype=row.dtype)
        colptr[1:] = paddle.cumsum(colcount, axis=0)

        col = paddle.arange(idx.shape[0]).to(row.place).repeat_interleave(colcount)

        perm = paddle.arange(col.shape[0]).to(col.place)
        perm += gather_csr(old_colptr[idx] - colptr[:-1], colptr)

        row = row[perm]
        csc2csr = (idx.shape[0] * row + col).argsort()
        row, col = row[csc2csr], col[csc2csr]

        if value is not None:
            value = value[perm][csc2csr]

        sparse_sizes = (src.sparse_size(0), idx.shape[0])

        storage = SparseStorage(
            row=row,
            rowptr=None,
            col=col,
            value=value,
            sparse_sizes=sparse_sizes,
            rowcount=None,
            colptr=colptr,
            colcount=colcount,
            csr2csc=None,
            csc2csr=csc2csr,
            is_sorted=True,
        )
        return src.from_storage(storage)

    else:
        value = src.storage.value()
        if value is not None:
            return src.set_value(value.index_select(dim - 1, idx), layout="coo")
        else:
            raise ValueError


def index_select_nnz(
    src: SparseTensor, idx: paddle.Tensor, layout: Optional[str] = None
) -> SparseTensor:
    assert idx.dim() == 1

    if get_layout(layout) == "csc":
        idx = src.storage.csc2csr()[idx]

    row, col, value = src.coo()
    row, col = row[idx], col[idx]

    if value is not None:
        value = value[idx]

    return SparseTensor(
        row=row,
        rowptr=None,
        col=col,
        value=value,
        sparse_sizes=src.sparse_sizes(),
        is_sorted=True,
    )


SparseTensor.index_select = lambda self, dim, idx: index_select(self, dim, idx)
tmp = lambda self, idx, layout=None: index_select_nnz(self, idx, layout)  # noqa
SparseTensor.index_select_nnz = tmp
