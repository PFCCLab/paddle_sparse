from __future__ import annotations

from textwrap import indent
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import scipy.sparse
from paddle_scatter import segment_csr

from paddle_sparse.storage import SparseStorage
from paddle_sparse.storage import get_layout


class SparseTensor(object):
    storage: SparseStorage

    def __init__(
        self,
        row: Optional[paddle.Tensor] = None,
        rowptr: Optional[paddle.Tensor] = None,
        col: Optional[paddle.Tensor] = None,
        value: Optional[paddle.Tensor] = None,
        sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
        is_sorted: bool = False,
        trust_data: bool = False,
    ):
        self.storage = SparseStorage(
            row=row,
            rowptr=rowptr,
            col=col,
            value=value,
            sparse_sizes=sparse_sizes,
            rowcount=None,
            colptr=None,
            colcount=None,
            csr2csc=None,
            csc2csr=None,
            is_sorted=is_sorted,
            trust_data=trust_data,
        )

    @classmethod
    def from_storage(self, storage: SparseStorage):
        out = SparseTensor(
            row=storage._row,
            rowptr=storage._rowptr,
            col=storage._col,
            value=storage._value,
            sparse_sizes=storage._sparse_sizes,
            is_sorted=True,
            trust_data=True,
        )
        out.storage._rowcount = storage._rowcount
        out.storage._colptr = storage._colptr
        out.storage._colcount = storage._colcount
        out.storage._csr2csc = storage._csr2csc
        out.storage._csc2csr = storage._csc2csr
        return out

    @classmethod
    def from_edge_index(
        self,
        edge_index: paddle.Tensor,
        edge_attr: Optional[paddle.Tensor] = None,
        sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
        is_sorted: bool = False,
        trust_data: bool = False,
    ):
        return SparseTensor(
            row=edge_index[0],
            rowptr=None,
            col=edge_index[1],
            value=edge_attr,
            sparse_sizes=sparse_sizes,
            is_sorted=is_sorted,
            trust_data=trust_data,
        )

    @classmethod
    def from_dense(self, mat: paddle.Tensor, has_value: bool = True):
        if mat.dim() > 2:
            index = mat.abs().sum([i for i in range(2, mat.dim())]).nonzero()
        else:
            index = mat.nonzero()
        index = index.t()

        row = index[0]
        col = index[1]

        value: Optional[paddle.Tensor] = None
        if has_value:
            value = mat[row, col]

        return SparseTensor(
            row=row,
            rowptr=None,
            col=col,
            value=value,
            sparse_sizes=(mat.shape[0], mat.shape[1]),
            is_sorted=True,
            trust_data=True,
        )

    @classmethod
    def from_paddle_sparse_coo_tensor(
        self,
        mat: paddle.Tensor,
        has_value: bool = True,
    ):
        mat = mat.coalesce()
        index = mat._indices()
        row, col = index[0], index[1]

        value: Optional[paddle.Tensor] = None
        if has_value:
            value = mat.values()

        return SparseTensor(
            row=row,
            rowptr=None,
            col=col,
            value=value,
            sparse_sizes=(mat.shape[0], mat.shape[1]),
            is_sorted=True,
            trust_data=True,
        )

    @classmethod
    def from_paddle_sparse_csr_tensor(
        self,
        mat: paddle.Tensor,
        has_value: bool = True,
    ):
        rowptr = mat.crows()
        col = mat.cols()

        value: Optional[paddle.Tensor] = None
        if has_value:
            value = mat.values()

        return SparseTensor(
            row=None,
            rowptr=rowptr,
            col=col,
            value=value,
            sparse_sizes=(mat.shape[0], mat.shape[1]),
            is_sorted=True,
            trust_data=True,
        )

    @classmethod
    def eye(
        self,
        M: int,
        N: Optional[int] = None,
        has_value: bool = True,
        dtype: Optional[int] = None,
        device: Optional[paddle.base.libpaddle.Place] = None,
        fill_cache: bool = False,
    ):

        N = M if N is None else N

        row = paddle.arange(min(M, N)).to(device)
        col = row

        rowptr = paddle.arange(M + 1).to(row.place)
        if M > N:
            rowptr[N + 1 :] = N

        value: Optional[paddle.Tensor] = None
        if has_value:
            value = paddle.ones([row.numel()], dtype=dtype).to(row.place)

        rowcount: Optional[paddle.Tensor] = None
        colptr: Optional[paddle.Tensor] = None
        colcount: Optional[paddle.Tensor] = None
        csr2csc: Optional[paddle.Tensor] = None
        csc2csr: Optional[paddle.Tensor] = None

        if fill_cache:
            rowcount = paddle.ones([M], dtype=paddle.int64).to(row.place)
            if M > N:
                rowcount[N:] = 0

            colptr = paddle.arange(N + 1, dtype=paddle.int64).to(row.place)
            colcount = paddle.ones([N], dtype=paddle.int64).to(row.place)
            if N > M:
                colptr[M + 1 :] = M
                colcount[M:] = 0
            csr2csc = csc2csr = row

        out = SparseTensor(
            row=row,
            rowptr=rowptr,
            col=col,
            value=value,
            sparse_sizes=(M, N),
            is_sorted=True,
            trust_data=True,
        )
        out.storage._rowcount = rowcount
        out.storage._colptr = colptr
        out.storage._colcount = colcount
        out.storage._csr2csc = csr2csc
        out.storage._csc2csr = csc2csr
        return out

    def copy(self):
        return self.from_storage(self.storage)

    def clone(self):
        return self.from_storage(self.storage.clone())

    def type(self, dtype: paddle.dtype, non_blocking: bool = False):
        value = self.storage.value()
        if value is None or dtype == value.dtype:
            return self
        return self.from_storage(
            self.storage.type(dtype=dtype, non_blocking=non_blocking)
        )

    def type_as(self, tensor: paddle.Tensor, non_blocking: bool = False):
        return self.type(dtype=tensor.dtype, non_blocking=non_blocking)

    def to_device(
        self, device: paddle.base.libpaddle.Place, non_blocking: bool = False
    ):
        if device == self.device():
            return self
        return self.from_storage(
            self.storage.to_device(device=device, non_blocking=non_blocking)
        )

    def device_as(self, tensor: paddle.Tensor, non_blocking: bool = False):
        return self.to_device(device=tensor.place, non_blocking=non_blocking)

    # Formats #################################################################

    def coo(self) -> Tuple[paddle.Tensor, paddle.Tensor, Optional[paddle.Tensor]]:
        return self.storage.row(), self.storage.col(), self.storage.value()

    def csr(self) -> Tuple[paddle.Tensor, paddle.Tensor, Optional[paddle.Tensor]]:
        return self.storage.rowptr(), self.storage.col(), self.storage.value()

    def csc(self) -> Tuple[paddle.Tensor, paddle.Tensor, Optional[paddle.Tensor]]:
        perm = self.storage.csr2csc()
        value = self.storage.value()
        if value is not None:
            value = value[perm]
        return self.storage.colptr(), self.storage.row()[perm], value

    # Storage inheritance #####################################################

    def has_value(self) -> bool:
        return self.storage.has_value()

    def set_value_(
        self,
        value: Optional[paddle.Tensor],
        layout: Optional[str] = None,
    ):
        self.storage.set_value_(value, layout)
        return self

    def set_value(
        self,
        value: Optional[paddle.Tensor],
        layout: Optional[str] = None,
    ):
        return self.from_storage(self.storage.set_value(value, layout))

    def sparse_sizes(self) -> Tuple[int, int]:
        return self.storage.sparse_sizes()

    def sparse_size(self, dim: int) -> int:
        return self.storage.sparse_sizes()[dim]

    def sparse_resize(self, sparse_sizes: Tuple[int, int]):
        return self.from_storage(self.storage.sparse_resize(sparse_sizes))

    def sparse_reshape(self, num_rows: int, num_cols: int):
        return self.from_storage(self.storage.sparse_reshape(num_rows, num_cols))

    def is_coalesced(self) -> bool:
        return self.storage.is_coalesced()

    def coalesce(self, reduce: str = "sum"):
        return self.from_storage(self.storage.coalesce(reduce))

    def fill_cache_(self):
        self.storage.fill_cache_()
        return self

    def clear_cache_(self):
        self.storage.clear_cache_()
        return self

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False

        if self.sizes() != other.sizes():
            return False

        rowptrA, colA, valueA = self.csr()
        rowptrB, colB, valueB = other.csr()

        if valueA is None and valueB is not None:
            return False
        if valueA is not None and valueB is None:
            return False
        if not paddle.equal_all(rowptrA, rowptrB):
            return False
        if not paddle.equal_all(colA, colB):
            return False
        if valueA is None and valueB is None:
            return True
        return paddle.equal_all(valueA, valueB)

    # Utility functions #######################################################

    def fill_value_(self, fill_value: float, dtype: Optional[int] = None):
        value = paddle.full(
            (self.nnz(),),
            fill_value,
            dtype=dtype,
        )
        return self.set_value_(value, layout="coo")

    def fill_value(self, fill_value: float, dtype: Optional[int] = None):
        value = paddle.full(
            (self.nnz(),),
            fill_value,
            dtype=dtype,
            device=self.device(),
        )
        return self.set_value(value, layout="coo")

    def sizes(self) -> List[int]:
        sparse_sizes = self.sparse_sizes()
        value = self.storage.value()
        if value is not None:
            return list(sparse_sizes) + list(value.shape)[1:]
        else:
            return list(sparse_sizes)

    def size(self, dim: int) -> int:
        return self.sizes()[dim]

    def dim(self) -> int:
        return len(self.sizes())

    def nnz(self) -> int:
        return self.storage.col().numel()

    def numel(self) -> int:
        value = self.storage.value()
        if value is not None:
            return value.numel()
        else:
            return self.nnz()

    def density(self) -> float:
        if self.sparse_size(0) == 0 or self.sparse_size(1) == 0:
            return 0.0
        return self.nnz() / (self.sparse_size(0) * self.sparse_size(1))

    def sparsity(self) -> float:
        return 1 - self.density()

    def avg_row_length(self) -> float:
        return self.nnz() / self.sparse_size(0)

    def avg_col_length(self) -> float:
        return self.nnz() / self.sparse_size(1)

    def bandwidth(self) -> int:
        row, col, _ = self.coo()
        return int((row - col).abs_().max())

    def avg_bandwidth(self) -> float:
        row, col, _ = self.coo()
        return float((row - col).abs_().astype(paddle.float32).mean())

    def bandwidth_proportion(self, bandwidth: int) -> float:
        row, col, _ = self.coo()
        tmp = (row - col).abs_()
        return int((tmp <= bandwidth).sum()) / self.nnz()

    def is_quadratic(self) -> bool:
        return self.sparse_size(0) == self.sparse_size(1)

    def is_symmetric(self) -> bool:
        if not self.is_quadratic():
            return False

        rowptr, col, value1 = self.csr()
        colptr, row, value2 = self.csc()

        if (rowptr != colptr).any() or (col != row).any():
            return False

        if value1 is None or value2 is None:
            return True
        else:
            return bool((value1 == value2).all())

    def to_symmetric(self, reduce: str = "sum"):
        N = max(self.size(0), self.size(1))

        row, col, value = self.coo()
        idx = paddle.full((2 * col.numel() + 1,), -1, dtype=col.dtype)
        idx[1 : row.numel() + 1] = row
        idx[row.numel() + 1 :] = col
        idx[1:] *= N
        idx[1 : row.numel() + 1] += col
        idx[row.numel() + 1 :] += row

        idx, perm = idx.sort(), idx.argsort()
        mask = idx[1:] > idx[:-1]
        perm = perm[1:].subtract_(paddle.full([], 1, dtype=perm.dtype))
        idx = perm[mask]

        if value is not None:
            ptr = mask.nonzero().flatten()
            ptr = paddle.concat(
                [ptr, paddle.full((1,), perm.shape[0], dtype=ptr.dtype)]
            )
            value = paddle.concat([value, value])[perm]
            value = segment_csr(value, ptr, reduce=reduce)

        new_row = paddle.concat([row, col], axis=0)[idx]
        new_col = paddle.concat([col, row], axis=0)[idx]

        out = SparseTensor(
            row=new_row,
            rowptr=None,
            col=new_col,
            value=value,
            sparse_sizes=(N, N),
            is_sorted=True,
            trust_data=True,
        )
        return out

    def detach_(self):
        value = self.storage.value()
        if value is not None:
            value.detach_()
        return self

    def detach(self):
        value = self.storage.value()
        if value is not None:
            value = value.detach()
        return self.set_value(value, layout="coo")

    def requires_grad(self) -> bool:
        value = self.storage.value()
        if value is not None:
            return not value.stop_gradient
        else:
            return False

    def requires_grad_(
        self,
        requires_grad: bool = True,
        dtype: Optional[int] = None,
    ):
        if requires_grad and not self.has_value():
            self.fill_value_(1.0, dtype)

        value = self.storage.value()
        if value is not None:
            value.stop_gradient = not requires_grad
        return self

    def pin_memory(self):
        return self.from_storage(self.storage.pin_memory())

    def is_pinned(self) -> bool:
        return self.storage.is_pinned()

    def device(self):
        return self.storage.col().place

    def cpu(self):
        return self.to_device(device=paddle.CPUPlace(), non_blocking=False)

    def cuda(self):
        return self.from_storage(self.storage.cuda())

    def is_cuda(self) -> bool:
        return self.storage.col().place.is_gpu_place()

    def dtype(self):
        value = self.storage.value()
        return value.dtype if value is not None else paddle.float32

    def is_floating_point(self) -> bool:
        value = self.storage.value()
        return paddle.is_floating_point(value) if value is not None else True

    def bfloat16(self):
        return self.type(dtype=paddle.bfloat16, non_blocking=False)

    def bool(self):
        return self.type(dtype=paddle.bool, non_blocking=False)

    def byte(self):
        return self.type(dtype=paddle.uint8, non_blocking=False)

    def char(self):
        return self.type(dtype=paddle.int8, non_blocking=False)

    def half(self):
        return self.type(dtype=paddle.float16, non_blocking=False)

    def float(self):
        return self.type(dtype=paddle.float32, non_blocking=False)

    def double(self):
        return self.type(dtype=paddle.float64, non_blocking=False)

    def short(self):
        return self.type(dtype=paddle.int16, non_blocking=False)

    def int(self):
        return self.type(dtype=paddle.int32, non_blocking=False)

    def long(self):
        return self.type(dtype=paddle.int64, non_blocking=False)

    # Conversions #############################################################

    def to_dense(self, dtype: Optional[int] = None) -> paddle.Tensor:
        row, col, value = self.coo()

        if value is not None:
            mat = paddle.zeros(
                self.sizes(),
                dtype=value.dtype,
            ).to(self.device())
        else:
            mat = paddle.zeros(self.sizes(), dtype=dtype).to(self.device())

        if value is not None:
            mat[row, col] = value
        else:
            mat[row, col] = paddle.ones(
                [self.nnz()],
                dtype=mat.dtype,
            ).to(mat.place)

        return mat

    def to_paddle_sparse_coo_tensor(
        self,
        dtype: Optional[int] = None,
    ) -> paddle.Tensor:
        row, col, value = self.coo()
        index = paddle.stack([row, col], axis=0)

        if value is None:
            value = paddle.ones([self.nnz()], dtype=dtype).to(self.device())

        return paddle.sparse.sparse_coo_tensor(index, value, self.sizes())

    def to_paddle_sparse_csr_tensor(
        self,
        dtype: Optional[int] = None,
    ) -> paddle.Tensor:
        rowptr, col, value = self.csr()

        if value is None:
            value = paddle.ones([self.nnz()], dtype=dtype).to(self.device())

        return paddle.sparse.sparse_csr_tensor(rowptr, col, value, self.sizes())

    def to_paddle_sparse_csc_tensor(
        self,
        dtype: Optional[int] = None,
    ) -> paddle.Tensor:
        raise NotImplementedError("Paddle not support sparse tensor with csc layout.")


# Python Bindings #############################################################


def share_memory_(self: SparseTensor) -> SparseTensor:
    self.storage.share_memory_()
    return self


def is_shared(self: SparseTensor) -> bool:
    return self.storage.is_shared()


def to(
    self, *args: Optional[List[Any]], **kwargs: Optional[Dict[str, Any]]
) -> SparseTensor:
    device = None
    dtype = None
    non_blocking = None
    size_args = len(args)
    size_kwargs = len(kwargs)

    def get_device_dtype_from_tensor(other):
        if other is not None:
            device = str(other.place)[6:-1]
            dtype = other.dtype
            return device, dtype
        else:
            return None, None

    if size_args + size_kwargs > 3 or size_args + size_kwargs == 0:
        raise TypeError(
            "to() received too many arguments - expected one of:\n  \
            * (Union[str, paddle.CPUPlace(), paddle.CUDAPlace(), paddle.CUDAPinnedPlace()] \
            device, Union[str, paddle.dtype, numpy.dtype] dtype, bool non_blocking)\n \
            * (Union[str, paddle.dtype, numpy.dtype] dtype, bool non_blocking)\n \
            * (paddle.Tensor other, bool non_blocking) "
        )
    valid_keys = {"device", "dtype", "blocking", "other"}
    valid_dtypes = [
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "complex64",
        "complex128",
        "bool",
    ]
    invalid_keys = set(kwargs.keys()) - valid_keys
    if len(invalid_keys) != 0:
        raise TypeError(
            "to() got an unexpected keyword argument " + next(iter(invalid_keys))
        )
    if size_args > 0:
        if isinstance(args[0], paddle.Tensor):
            device, dtype = get_device_dtype_from_tensor(args[0])
            if size_args == 2:
                non_blocking = args[1]
            else:
                non_blocking = kwargs.get("non_blocking", None)
        elif (
            isinstance(args[0], (paddle.dtype, np.dtype))
            or isinstance(args[0], str)
            and args[0].lower() in valid_dtypes
        ):
            dtype = args[0]
            if size_args == 2:
                non_blocking = args[1]
            else:
                non_blocking = kwargs.get("non_blocking", None)
        else:
            device = args[0]
            if size_args == 2:
                dtype = args[1]
            elif size_args == 3:
                dtype, non_blocking = args[1], args[2]
            else:
                dtype = kwargs.get("dtype", None)
                non_blocking = kwargs.get("non_blocking", None)
    else:
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        non_blocking = kwargs.get("non_blocking", None)
        if device is None and dtype is None:
            device, dtype = get_device_dtype_from_tensor(kwargs.get("other", None))
    non_blocking = False if non_blocking is None else True
    if dtype is not None:
        self = self.type(dtype=dtype, non_blocking=non_blocking)
    if device is not None:
        self = self.to_device(device=device, non_blocking=non_blocking)

    return self


def cpu(self) -> SparseTensor:
    return self.device_as(paddle.to_tensor(0.0, place="cpu"))


def cuda(
    self,
    device: Optional[Union[int, str]] = None,
    non_blocking: bool = False,
):
    return self.device_as(paddle.to_tensor(0.0, place=device or "gpu"), non_blocking)


def __getitem__(self: SparseTensor, index: Any) -> SparseTensor:
    index = list(index) if isinstance(index, tuple) else [index]
    # More than one `Ellipsis` is not allowed...
    if (
        len(
            [
                i
                for i in index
                if not isinstance(i, (paddle.Tensor, np.ndarray)) and i == ...
            ]
        )
        > 1
    ):
        raise SyntaxError

    dim = 0
    out = self
    while len(index) > 0:
        item = index.pop(0)
        if isinstance(item, (list, tuple)):
            item = paddle.to_tensor(item, place=self.device())
        if isinstance(item, np.ndarray):
            item = paddle.to_tensor(item).to(self.device())

        if isinstance(item, int):
            out = out.select(dim, item)
            dim += 1
        elif isinstance(item, slice):
            if item.step is not None:
                raise ValueError("Step parameter not yet supported.")

            start = 0 if item.start is None else item.start
            start = self.size(dim) + start if start < 0 else start

            stop = self.size(dim) if item.stop is None else item.stop
            stop = self.size(dim) + stop if stop < 0 else stop

            out = out.narrow(dim, start, max(stop - start, 0))
            dim += 1
        elif paddle.is_tensor(item):
            if item.dtype == paddle.bool:
                out = out.masked_select(dim, item)
                dim += 1
            elif item.dtype == paddle.int64:
                out = out.index_select(dim, item)
                dim += 1
        elif item == Ellipsis:
            if self.dim() - len(index) < dim:
                raise SyntaxError
            dim = self.dim() - len(index)
        else:
            raise SyntaxError

    return out


def __repr__(self: SparseTensor) -> str:
    i = " " * 6
    row, col, value = self.coo()
    infos = []
    infos += [f"row={indent(row.__repr__(), i)[len(i):]}"]
    infos += [f"col={indent(col.__repr__(), i)[len(i):]}"]

    if value is not None:
        infos += [f"val={indent(value.__repr__(), i)[len(i):]}"]

    infos += [
        f"size={tuple(self.sizes())}, nnz={self.nnz()}, "
        f"density={100 * self.density():.02f}%"
    ]

    infos = ",\n".join(infos)

    i = " " * (len(self.__class__.__name__) + 1)
    return f"{self.__class__.__name__}({indent(infos, i)[len(i):]})"


SparseTensor.share_memory_ = share_memory_
SparseTensor.is_shared = is_shared
SparseTensor.to = to
SparseTensor.cpu = cpu
SparseTensor.cuda = cuda
SparseTensor.__getitem__ = __getitem__
SparseTensor.__repr__ = __repr__

# Scipy Conversions ###########################################################

ScipySparseMatrix = Union[
    scipy.sparse.coo_matrix, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix
]


def from_scipy(mat: ScipySparseMatrix, has_value: bool = True) -> SparseTensor:
    colptr = None
    if isinstance(mat, scipy.sparse.csc_matrix):
        colptr = paddle.to_tensor(mat.indptr).to(paddle.int64)

    mat = mat.tocsr()
    rowptr = paddle.to_tensor(mat.indptr).to(paddle.int64)
    mat = mat.tocoo()
    row = paddle.to_tensor(mat.row).to(paddle.int64)
    col = paddle.to_tensor(mat.col).to(paddle.int64)
    value = None
    if has_value:
        value = paddle.to_tensor(mat.data)
    sparse_sizes = mat.shape[:2]

    storage = SparseStorage(
        row=row,
        rowptr=rowptr,
        col=col,
        value=value,
        sparse_sizes=sparse_sizes,
        rowcount=None,
        colptr=colptr,
        colcount=None,
        csr2csc=None,
        csc2csr=None,
        is_sorted=True,
    )

    return SparseTensor.from_storage(storage)


def to_scipy(
    self: SparseTensor,
    layout: Optional[str] = None,
    dtype: Optional[paddle.dtype] = None,
) -> ScipySparseMatrix:
    assert self.dim() == 2
    layout = get_layout(layout)

    if not self.has_value():
        ones = paddle.ones(self.nnz(), dtype=dtype).numpy()

    if layout == "coo":
        row, col, value = self.coo()
        row = row.detach().cpu().numpy()
        col = col.detach().cpu().numpy()
        value = value.detach().cpu().numpy() if self.has_value() else ones
        return scipy.sparse.coo_matrix((value, (row, col)), self.sizes())
    elif layout == "csr":
        rowptr, col, value = self.csr()
        rowptr = rowptr.detach().cpu().numpy()
        col = col.detach().cpu().numpy()
        value = value.detach().cpu().numpy() if self.has_value() else ones
        return scipy.sparse.csr_matrix((value, col, rowptr), self.sizes())
    elif layout == "csc":
        colptr, row, value = self.csc()
        colptr = colptr.detach().cpu().numpy()
        row = row.detach().cpu().numpy()
        value = value.detach().cpu().numpy() if self.has_value() else ones
        return scipy.sparse.csc_matrix((value, row, colptr), self.sizes())


SparseTensor.from_scipy = from_scipy
SparseTensor.to_scipy = to_scipy
