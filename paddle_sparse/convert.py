import numpy as np
import paddle
import scipy.sparse
from paddle import to_tensor


def to_paddle_sparse(index, value, m, n):
    return paddle.sparse.sparse_coo_tensor(index.detach(), value, (m, n))


def from_paddle_sparse(A):
    return A.indices().detach(), A.values()


def to_scipy(index, value, m, n):
    assert not index.place.is_gpu_place() and not value.place.is_gpu_place()
    (row, col), data = index.detach(), value.detach()
    return scipy.sparse.coo_matrix((data, (row, col)), (m, n))


def from_scipy(A):
    A = A.tocoo()
    row, col, value = A.row.astype(np.int64), A.col.astype(np.int64), A.data
    row, col, value = to_tensor(row), to_tensor(col), to_tensor(value)
    index = paddle.stack([row, col], axis=0)
    return index, value
