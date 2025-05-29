from itertools import product

import numpy as np
import paddle
import pytest

from paddle_sparse import SparseTensor
from paddle_sparse.testing import devices
from paddle_sparse.testing import grad_dtypes

np.random.seed(1234)


@pytest.mark.parametrize("dtype,device", product(grad_dtypes, devices))
def test_getitem(dtype, device):
    device = str(device)[6:-1]
    paddle.device.set_device(device)

    m = 50
    n = 40
    k = 10
    mat = paddle.randn([m, n], dtype=dtype)
    mat = SparseTensor.from_dense(mat)

    # idx1 = paddle.to_tensor(np.random.randint(0, m, (k, )), dtype=paddle.int64)
    # idx2 = paddle.to_tensor(np.random.randint(0, n, (k, )), dtype=paddle.int64)

    idx1 = paddle.randint(0, m, (k,), dtype=paddle.int64)
    idx2 = paddle.randint(0, n, (k,), dtype=paddle.int64)
    bool1 = paddle.zeros([m], dtype=paddle.bool)
    bool2 = paddle.zeros([n], dtype=paddle.bool)
    bool1 = bool1.astype(paddle.int64).put_along_axis_(idx1, 1, 0).astype(paddle.bool)
    bool2 = bool2.astype(paddle.int64).put_along_axis_(idx2, 1, 0).astype(paddle.bool)

    # idx1 and idx2 may have duplicates
    k1_bool = bool1.nonzero().shape[0]
    k2_bool = bool2.nonzero().shape[0]

    idx1np = idx1.cpu().numpy()
    idx2np = idx2.cpu().numpy()
    bool1np = bool1.cpu().numpy()
    bool2np = bool2.cpu().numpy()

    idx1list = idx1np.tolist()
    idx2list = idx2np.tolist()
    bool1list = bool1np.tolist()
    bool2list = bool2np.tolist()

    assert mat[:k, :k].sizes() == [k, k]
    assert mat[..., :k].sizes() == [m, k]

    assert mat[idx1, idx2].sizes() == [k, k]
    assert mat[idx1np, idx2np].sizes() == [k, k]
    assert mat[idx1list, idx2list].sizes() == [k, k]

    assert mat[bool1, bool2].sizes() == [k1_bool, k2_bool]
    assert mat[bool1np, bool2np].sizes() == [k1_bool, k2_bool]
    assert mat[bool1list, bool2list].sizes() == [k1_bool, k2_bool]

    assert mat[idx1].sizes() == [k, n]
    assert mat[idx1np].sizes() == [k, n]
    assert mat[idx1list].sizes() == [k, n]

    assert mat[bool1].sizes() == [k1_bool, n]
    assert mat[bool1np].sizes() == [k1_bool, n]
    assert mat[bool1list].sizes() == [k1_bool, n]


@pytest.mark.parametrize("device", devices)
def test_to_symmetric(device):
    device = str(device)[6:-1]
    paddle.device.set_device(str(device))

    row = paddle.to_tensor([0, 0, 0, 1, 1])
    col = paddle.to_tensor([0, 1, 2, 0, 2])
    value = paddle.arange(1, 6)
    mat = SparseTensor(row=row, col=col, value=value)
    assert not mat.is_symmetric()

    mat = mat.to_symmetric()

    assert mat.is_symmetric()
    assert mat.to_dense().tolist() == [
        [2, 6, 3],
        [6, 0, 5],
        [3, 5, 0],
    ]


def test_equal():
    row = paddle.to_tensor([0, 0, 0, 1, 1])
    col = paddle.to_tensor([0, 1, 2, 0, 2])
    value = paddle.arange(1, 6)
    matA = SparseTensor(row=row, col=col, value=value)
    matB = SparseTensor(row=row, col=col, value=value)
    col = paddle.to_tensor([0, 1, 2, 0, 1])
    matC = SparseTensor(row=row, col=col, value=value)

    assert id(matA) != id(matB)
    assert matA == matB

    assert id(matA) != id(matC)
    assert matA != matC


def test_to():
    row = paddle.to_tensor([0, 0, 0, 1, 1])
    col = paddle.to_tensor([0, 1, 2, 0, 2])
    value = paddle.arange(1, 6)
    mat = SparseTensor(row=row, col=col, value=value)

    assert value.dtype == paddle.int64

    mat = mat.to(paddle.float32)

    assert mat.storage.value().dtype == paddle.float32

    mat = mat.to(paddle.CPUPlace(), paddle.float32)

    assert str(mat.storage.value().place) == str(paddle.CPUPlace())
    assert str(mat.storage.row().place) == str(paddle.CPUPlace())
    assert str(mat.storage.col().place) == str(paddle.CPUPlace())
