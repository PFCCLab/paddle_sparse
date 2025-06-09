from __future__ import annotations

import warnings
from typing import List
from typing import Optional
from typing import Tuple

import paddle
import paddle_sparse_ops
from paddle_scatter import scatter_add
from paddle_scatter import segment_csr

from paddle_sparse.utils import Final
from paddle_sparse.utils import index_sort
from paddle_sparse.utils import is_pinned_tensor

layouts: Final[List[str]] = ["coo", "csr", "csc"]


def get_layout(layout: Optional[str] = None) -> str:
    if layout is None:
        layout = "coo"
        warnings.warn(
            "`layout` argument unset, using default layout "
            '"coo". This may lead to unexpected behaviour.'
        )
    assert layout == "coo" or layout == "csr" or layout == "csc"
    return layout


class SparseStorage(object):
    _row: Optional[paddle.Tensor]
    _rowptr: Optional[paddle.Tensor]
    _col: paddle.Tensor
    _value: Optional[paddle.Tensor]
    _sparse_sizes: Tuple[int, int]
    _rowcount: Optional[paddle.Tensor]
    _colptr: Optional[paddle.Tensor]
    _colcount: Optional[paddle.Tensor]
    _csr2csc: Optional[paddle.Tensor]
    _csc2csr: Optional[paddle.Tensor]

    def __init__(
        self,
        row: Optional[paddle.Tensor] = None,
        rowptr: Optional[paddle.Tensor] = None,
        col: Optional[paddle.Tensor] = None,
        value: Optional[paddle.Tensor] = None,
        sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
        rowcount: Optional[paddle.Tensor] = None,
        colptr: Optional[paddle.Tensor] = None,
        colcount: Optional[paddle.Tensor] = None,
        csr2csc: Optional[paddle.Tensor] = None,
        csc2csr: Optional[paddle.Tensor] = None,
        is_sorted: bool = False,
        trust_data: bool = False,
    ):

        assert row is not None or rowptr is not None
        assert col is not None
        assert col.dtype == paddle.int64
        assert col.dim() == 1
        col = col.contiguous()

        M: int = 0
        if sparse_sizes is None or sparse_sizes[0] is None:
            if rowptr is not None:
                M = rowptr.numel() - 1
            elif row is not None and row.numel() > 0:
                M = int(row.max()) + 1
        else:
            _M = sparse_sizes[0]
            assert _M is not None
            M = _M
            if rowptr is not None:
                assert rowptr.numel() - 1 == M
            elif row is not None and row.numel() > 0:
                assert trust_data or int(row.max()) < M

        N: int = 0
        if sparse_sizes is None or sparse_sizes[1] is None:
            if col.numel() > 0:
                N = int(col.max()) + 1
        else:
            _N = sparse_sizes[1]
            assert _N is not None
            N = _N
            if col.numel() > 0:
                assert trust_data or int(col.max()) < N

        sparse_sizes = (M, N)

        if row is not None:
            assert row.dtype == paddle.int64
            assert row.place == col.place
            assert row.dim() == 1
            assert row.numel() == col.numel()
            row = row.contiguous()

        if rowptr is not None:
            assert rowptr.dtype == paddle.int64
            assert rowptr.place == col.place
            assert rowptr.dim() == 1
            assert rowptr.numel() - 1 == sparse_sizes[0]
            rowptr = rowptr.contiguous()

        if value is not None:
            assert value.place == col.place
            assert value.shape[0] == col.shape[0]
            value = value.contiguous()

        if rowcount is not None:
            assert rowcount.dtype == paddle.int64
            assert rowcount.place == col.place
            assert rowcount.dim() == 1
            assert rowcount.numel() == sparse_sizes[0]
            rowcount = rowcount.contiguous()

        if colptr is not None:
            assert colptr.dtype == paddle.int64
            assert colptr.place == col.place
            assert colptr.dim() == 1
            assert colptr.numel() - 1 == sparse_sizes[1]
            colptr = colptr.contiguous()

        if colcount is not None:
            assert colcount.dtype == paddle.int64
            assert colcount.place == col.place
            assert colcount.dim() == 1
            assert colcount.numel() == sparse_sizes[1]
            colcount = colcount.contiguous()

        if csr2csc is not None:
            assert csr2csc.dtype == paddle.int64
            assert csr2csc.place == col.place
            assert csr2csc.dim() == 1
            assert csr2csc.numel() == col.shape[0]
            csr2csc = csr2csc.contiguous()

        if csc2csr is not None:
            assert csc2csr.dtype == paddle.int64
            assert csc2csr.place == col.place
            assert csc2csr.dim() == 1
            assert csc2csr.numel() == col.shape[0]
            csc2csr = csc2csr.contiguous()

        self._row = row
        self._rowptr = rowptr
        self._col = col
        self._value = value
        self._sparse_sizes = tuple(sparse_sizes)
        self._rowcount = rowcount
        self._colptr = colptr
        self._colcount = colcount
        self._csr2csc = csr2csc
        self._csc2csr = csc2csr

        if not is_sorted and self._col.numel() > 0:
            idx = paddle.zeros((self._col.numel() + 1), dtype=self._col.dtype)
            idx[1:] = self.row()
            idx[1:] *= self._sparse_sizes[1]
            idx[1:] += self._col
            if (idx[1:] < idx[:-1]).any():
                max_value = self._sparse_sizes[0] * self._sparse_sizes[1]
                _, perm = index_sort(idx[1:], max_value)
                self._row = self.row()[perm]
                self._col = self._col[perm]
                if value is not None:
                    self._value = value[perm]
                self._csr2csc = None
                self._csc2csr = None

    @classmethod
    def empty(self):
        row = paddle.tensor([], dtype=paddle.int64)
        col = paddle.tensor([], dtype=paddle.int64)
        return SparseStorage(
            row=row,
            rowptr=None,
            col=col,
            value=None,
            sparse_sizes=(0, 0),
            rowcount=None,
            colptr=None,
            colcount=None,
            csr2csc=None,
            csc2csr=None,
            is_sorted=True,
            trust_data=True,
        )

    def has_row(self) -> bool:
        return self._row is not None

    def row(self):
        row = self._row
        if row is not None:
            return row

        rowptr = self._rowptr
        if rowptr is not None:
            row = paddle_sparse_ops.ptr2ind(rowptr, self._col.numel())
            self._row = row
            return row

        raise ValueError

    def has_rowptr(self) -> bool:
        return self._rowptr is not None

    def rowptr(self) -> paddle.Tensor:
        rowptr = self._rowptr
        if rowptr is not None:
            return rowptr

        row = self._row
        if row is not None:
            rowptr = paddle_sparse_ops.ind2ptr(row, self._sparse_sizes[0])
            self._rowptr = rowptr
            return rowptr

        raise ValueError

    def col(self) -> paddle.Tensor:
        return self._col

    def has_value(self) -> bool:
        return self._value is not None

    def value(self) -> Optional[paddle.Tensor]:
        return self._value

    def set_value_(
        self,
        value: Optional[paddle.Tensor],
        layout: Optional[str] = None,
    ):
        if value is not None:
            if get_layout(layout) == "csc":
                value = value[self.csc2csr()]
            value = value.contiguous()
            assert value.place == self._col.place
            assert value.shape[0] == self._col.numel()

        self._value = value
        return self

    def set_value(
        self,
        value: Optional[paddle.Tensor],
        layout: Optional[str] = None,
    ):
        if value is not None:
            if get_layout(layout) == "csc":
                value = value[self.csc2csr()]
            value = value.contiguous()
            assert value.place == self._col.place
            assert value.shape[0] == self._col.numel()

        return SparseStorage(
            row=self._row,
            rowptr=self._rowptr,
            col=self._col,
            value=value,
            sparse_sizes=self._sparse_sizes,
            rowcount=self._rowcount,
            colptr=self._colptr,
            colcount=self._colcount,
            csr2csc=self._csr2csc,
            csc2csr=self._csc2csr,
            is_sorted=True,
            trust_data=True,
        )

    def sparse_sizes(self) -> Tuple[int, int]:
        return self._sparse_sizes

    def sparse_size(self, dim: int) -> int:
        return self._sparse_sizes[dim]

    def sparse_resize(self, sparse_sizes: Tuple[int, int]):
        assert len(sparse_sizes) == 2
        old_sparse_sizes, nnz = self._sparse_sizes, self._col.numel()

        diff_0 = sparse_sizes[0] - old_sparse_sizes[0]
        rowcount, rowptr = self._rowcount, self._rowptr
        if diff_0 > 0:
            if rowptr is not None:
                rowptr = paddle.concat(
                    [rowptr, paddle.full((diff_0,), nnz, dtype=rowptr.dtype)]
                )
            if rowcount is not None:
                rowcount = paddle.concat(
                    [rowcount, paddle.zeros(diff_0, dtype=rowcount.dtype)]
                )
        elif diff_0 < 0:
            if rowptr is not None:
                rowptr = rowptr[:diff_0]
            if rowcount is not None:
                rowcount = rowcount[:diff_0]

        diff_1 = sparse_sizes[1] - old_sparse_sizes[1]
        colcount, colptr = self._colcount, self._colptr
        if diff_1 > 0:
            if colptr is not None:
                colptr = paddle.concat(
                    [colptr, paddle.full((diff_1,), nnz, dtype=colptr.dtype)]
                )
            if colcount is not None:
                colcount = paddle.concat(
                    [colcount, paddle.zeros(diff_1, dtype=colcount.dtype)]
                )
        elif diff_1 < 0:
            if colptr is not None:
                colptr = colptr[:diff_1]
            if colcount is not None:
                colcount = colcount[:diff_1]

        return SparseStorage(
            row=self._row,
            rowptr=rowptr,
            col=self._col,
            value=self._value,
            sparse_sizes=sparse_sizes,
            rowcount=rowcount,
            colptr=colptr,
            colcount=colcount,
            csr2csc=self._csr2csc,
            csc2csr=self._csc2csr,
            is_sorted=True,
            trust_data=True,
        )

    def sparse_reshape(self, num_rows: int, num_cols: int):
        assert num_rows > 0 or num_rows == -1
        assert num_cols > 0 or num_cols == -1
        assert num_rows > 0 or num_cols > 0

        total = self.sparse_size(0) * self.sparse_size(1)

        if num_rows == -1:
            num_rows = total // num_cols

        if num_cols == -1:
            num_cols = total // num_rows

        assert num_rows * num_cols == total

        idx = self.sparse_size(1) * self.row() + self.col()

        row = paddle.floor_divide(idx, paddle.full([], num_cols, dtype=idx.dtype))
        col = idx % num_cols
        assert row.dtype == paddle.int64 and col.dtype == paddle.int64

        return SparseStorage(
            row=row,
            rowptr=None,
            col=col,
            value=self._value,
            sparse_sizes=(num_rows, num_cols),
            rowcount=None,
            colptr=None,
            colcount=None,
            csr2csc=None,
            csc2csr=None,
            is_sorted=True,
            trust_data=True,
        )

    def has_rowcount(self) -> bool:
        return self._rowcount is not None

    def rowcount(self) -> paddle.Tensor:
        rowcount = self._rowcount
        if rowcount is not None:
            return rowcount

        rowptr = self.rowptr()
        rowcount = rowptr[1:] - rowptr[:-1]
        self._rowcount = rowcount
        return rowcount

    def has_colptr(self) -> bool:
        return self._colptr is not None

    def colptr(self) -> paddle.Tensor:
        colptr = self._colptr
        if colptr is not None:
            return colptr

        csr2csc = self._csr2csc
        if csr2csc is not None:
            colptr = paddle_sparse_ops.ind2ptr(
                self._col[csr2csc], self._sparse_sizes[1]
            )
        else:
            colptr = paddle.zeros((self._sparse_sizes[1] + 1), dtype=self._col.dtype)
            colptr[1:] = paddle.cumsum(self.colcount(), axis=0)
        self._colptr = colptr
        return colptr

    def has_colcount(self) -> bool:
        return self._colcount is not None

    def colcount(self) -> paddle.Tensor:
        colcount = self._colcount
        if colcount is not None:
            return colcount

        colptr = self._colptr
        if colptr is not None:
            colcount = colptr[1:] - colptr[:-1]
        else:
            colcount = scatter_add(
                paddle.ones_like(self._col),
                self._col,
                dim_size=self._sparse_sizes[1],
            )
        self._colcount = colcount
        return colcount

    def has_csr2csc(self) -> bool:
        return self._csr2csc is not None

    def csr2csc(self) -> paddle.Tensor:
        csr2csc = self._csr2csc
        if csr2csc is not None:
            return csr2csc

        idx = self._sparse_sizes[0] * self._col + self.row()
        max_value = self._sparse_sizes[0] * self._sparse_sizes[1]
        _, csr2csc = index_sort(idx, max_value)
        self._csr2csc = csr2csc
        return csr2csc

    def has_csc2csr(self) -> bool:
        return self._csc2csr is not None

    def csc2csr(self) -> paddle.Tensor:
        csc2csr = self._csc2csr
        if csc2csr is not None:
            return csc2csr

        max_value = self._sparse_sizes[0] * self._sparse_sizes[1]
        _, csc2csr = index_sort(self.csr2csc(), max_value)
        self._csc2csr = csc2csr
        return csc2csr

    def is_coalesced(self) -> bool:
        idx = paddle.full((self._col.numel() + 1,), -1, dtype=self._col.dtype)
        idx[1:] = self._sparse_sizes[1] * self.row() + self._col
        return bool((idx[1:] > idx[:-1]).all())

    def coalesce(self, reduce: str = "add"):
        idx = paddle.full((self._col.numel() + 1,), -1, dtype=self._col.dtype)
        idx[1:] = self._sparse_sizes[1] * self.row() + self._col
        mask = idx[1:] > idx[:-1]

        if mask.all():  # Skip if indices are already coalesced.
            return self

        row = self.row()[mask]
        col = self._col[mask]

        value = self._value
        if value is not None:
            ptr = mask.nonzero().flatten()
            ptr = paddle.concat(
                [ptr, paddle.full((1,), value.shape[0], dtype=ptr.dtype)]
            )
            value = segment_csr(value, ptr, reduce=reduce)

        return SparseStorage(
            row=row,
            rowptr=None,
            col=col,
            value=value,
            sparse_sizes=self._sparse_sizes,
            rowcount=None,
            colptr=None,
            colcount=None,
            csr2csc=None,
            csc2csr=None,
            is_sorted=True,
            trust_data=True,
        )

    def fill_cache_(self):
        self.row()
        self.rowptr()
        self.rowcount()
        self.colptr()
        self.colcount()
        self.csr2csc()
        self.csc2csr()
        return self

    def clear_cache_(self):
        self._rowcount = None
        self._colptr = None
        self._colcount = None
        self._csr2csc = None
        self._csc2csr = None
        return self

    def cached_keys(self) -> List[str]:
        keys: List[str] = []
        if self.has_rowcount():
            keys.append("rowcount")
        if self.has_colptr():
            keys.append("colptr")
        if self.has_colcount():
            keys.append("colcount")
        if self.has_csr2csc():
            keys.append("csr2csc")
        if self.has_csc2csr():
            keys.append("csc2csr")
        return keys

    def num_cached_keys(self) -> int:
        return len(self.cached_keys())

    def copy(self):
        return SparseStorage(
            row=self._row,
            rowptr=self._rowptr,
            col=self._col,
            value=self._value,
            sparse_sizes=self._sparse_sizes,
            rowcount=self._rowcount,
            colptr=self._colptr,
            colcount=self._colcount,
            csr2csc=self._csr2csc,
            csc2csr=self._csc2csr,
            is_sorted=True,
            trust_data=True,
        )

    def clone(self):
        row = self._row
        if row is not None:
            row = row.clone()
        rowptr = self._rowptr
        if rowptr is not None:
            rowptr = rowptr.clone()
        col = self._col.clone()
        value = self._value
        if value is not None:
            value = value.clone()
        rowcount = self._rowcount
        if rowcount is not None:
            rowcount = rowcount.clone()
        colptr = self._colptr
        if colptr is not None:
            colptr = colptr.clone()
        colcount = self._colcount
        if colcount is not None:
            colcount = colcount.clone()
        csr2csc = self._csr2csc
        if csr2csc is not None:
            csr2csc = csr2csc.clone()
        csc2csr = self._csc2csr
        if csc2csr is not None:
            csc2csr = csc2csr.clone()

        return SparseStorage(
            row=row,
            rowptr=rowptr,
            col=col,
            value=value,
            sparse_sizes=self._sparse_sizes,
            rowcount=rowcount,
            colptr=colptr,
            colcount=colcount,
            csr2csc=csr2csc,
            csc2csr=csc2csr,
            is_sorted=True,
            trust_data=True,
        )

    def type(self, dtype: paddle.dtype, non_blocking: bool = False):
        value = self._value
        if value is not None:
            if dtype == value.dtype:
                return self
            else:
                return self.set_value(
                    value.to(dtype=dtype, blocking=not non_blocking),
                    layout="coo",
                )
        else:
            return self

    def type_as(self, tensor: paddle.Tensor, non_blocking: bool = False):
        return self.type(dtype=tensor.dtype, non_blocking=non_blocking)

    def to_device(
        self, device: paddle.base.libpaddle.Place, non_blocking: bool = False
    ):
        if device == self._col.place:
            return self

        blocking = not non_blocking
        row = self._row
        if row is not None:
            row = row.to(device, blocking=blocking).astype(row.dtype)
        rowptr = self._rowptr
        if rowptr is not None:
            rowptr = rowptr.to(device, blocking=blocking).astype(rowptr.dtype)
        col = self._col.to(device, blocking=blocking).astype(self._col.dtype)
        value = self._value
        if value is not None:
            value = value.to(device, blocking=blocking).astype(value.dtype)
        rowcount = self._rowcount
        if rowcount is not None:
            rowcount = rowcount.to(device, blocking=blocking).astype(rowcount.dtype)
        colptr = self._colptr
        if colptr is not None:
            colptr = colptr.to(device, blocking=blocking).astype(colptr.dtype)
        colcount = self._colcount
        if colcount is not None:
            colcount = colcount.to(device, blocking=blocking).astype(colcount.dtype)
        csr2csc = self._csr2csc
        if csr2csc is not None:
            csr2csc = csr2csc.to(device, blocking=blocking).astype(csr2csc.dtype)
        csc2csr = self._csc2csr
        if csc2csr is not None:
            csc2csr = csc2csr.to(device, blocking=blocking).astype(csc2csr.dtype)

        return SparseStorage(
            row=row,
            rowptr=rowptr,
            col=col,
            value=value,
            sparse_sizes=self._sparse_sizes,
            rowcount=rowcount,
            colptr=colptr,
            colcount=colcount,
            csr2csc=csr2csc,
            csc2csr=csc2csr,
            is_sorted=True,
            trust_data=True,
        )

    def device_as(self, tensor: paddle.Tensor, non_blocking: bool = False):
        return self.to_device(device=tensor.place, non_blocking=non_blocking)

    def cuda(self):
        new_col = self._col.cuda()
        if new_col.place == self._col.place:
            return self

        row = self._row
        if row is not None:
            row = row.cuda()
        rowptr = self._rowptr
        if rowptr is not None:
            rowptr = rowptr.cuda()
        value = self._value
        if value is not None:
            value = value.cuda()
        rowcount = self._rowcount
        if rowcount is not None:
            rowcount = rowcount.cuda()
        colptr = self._colptr
        if colptr is not None:
            colptr = colptr.cuda()
        colcount = self._colcount
        if colcount is not None:
            colcount = colcount.cuda()
        csr2csc = self._csr2csc
        if csr2csc is not None:
            csr2csc = csr2csc.cuda()
        csc2csr = self._csc2csr
        if csc2csr is not None:
            csc2csr = csc2csr.cuda()

        return SparseStorage(
            row=row,
            rowptr=rowptr,
            col=new_col,
            value=value,
            sparse_sizes=self._sparse_sizes,
            rowcount=rowcount,
            colptr=colptr,
            colcount=colcount,
            csr2csc=csr2csc,
            csc2csr=csc2csr,
            is_sorted=True,
            trust_data=True,
        )

    def pin_memory(self):
        row = self._row
        if row is not None:
            row = row.pin_memory()
        rowptr = self._rowptr
        if rowptr is not None:
            rowptr = rowptr.pin_memory()
        col = self._col.pin_memory()
        value = self._value
        if value is not None:
            value = value.pin_memory()
        rowcount = self._rowcount
        if rowcount is not None:
            rowcount = rowcount.pin_memory()
        colptr = self._colptr
        if colptr is not None:
            colptr = colptr.pin_memory()
        colcount = self._colcount
        if colcount is not None:
            colcount = colcount.pin_memory()
        csr2csc = self._csr2csc
        if csr2csc is not None:
            csr2csc = csr2csc.pin_memory()
        csc2csr = self._csc2csr
        if csc2csr is not None:
            csc2csr = csc2csr.pin_memory()

        return SparseStorage(
            row=row,
            rowptr=rowptr,
            col=col,
            value=value,
            sparse_sizes=self._sparse_sizes,
            rowcount=rowcount,
            colptr=colptr,
            colcount=colcount,
            csr2csc=csr2csc,
            csc2csr=csc2csr,
            is_sorted=True,
            trust_data=True,
        )

    def is_pinned(self) -> bool:
        is_pinned = True
        row = self._row
        if row is not None:
            is_pinned = is_pinned and is_pinned(row)
        rowptr = self._rowptr
        if rowptr is not None:
            is_pinned = is_pinned and is_pinned_tensor(rowptr)
        is_pinned = is_pinned_tensor(self._col)
        value = self._value
        if value is not None:
            is_pinned = is_pinned and is_pinned_tensor(value)
        rowcount = self._rowcount
        if rowcount is not None:
            is_pinned = is_pinned and is_pinned_tensor(rowcount)
        colptr = self._colptr
        if colptr is not None:
            is_pinned = is_pinned and is_pinned_tensor(colptr)
        colcount = self._colcount
        if colcount is not None:
            is_pinned = is_pinned and is_pinned_tensor(colcount)
        csr2csc = self._csr2csc
        if csr2csc is not None:
            is_pinned = is_pinned and is_pinned_tensor(csr2csc)
        csc2csr = self._csc2csr
        if csc2csr is not None:
            is_pinned = is_pinned and is_pinned_tensor(csc2csr)
        return is_pinned


def share_memory_(self) -> SparseStorage:
    warnings.warn("Paddle not support `shard_memory_`. Please avoid call this api.")


def is_shared(self) -> bool:
    warnings.warn("Paddle not support `is_shared`. The return value is always `False`.")

    return False


SparseStorage.share_memory_ = share_memory_
SparseStorage.is_shared = is_shared
