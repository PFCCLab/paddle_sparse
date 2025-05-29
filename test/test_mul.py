from itertools import product

import paddle
import pytest

from paddle_sparse import SparseTensor
from paddle_sparse import mul
from paddle_sparse.testing import devices
from paddle_sparse.testing import dtypes
from paddle_sparse.testing import tensor


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_sparse_sparse_mul(dtype, device):
    device = str(device)[6:-1]
    paddle.device.set_device(device)

    rowA = paddle.to_tensor([0, 0, 1, 2, 2])
    colA = paddle.to_tensor([0, 2, 1, 0, 1])
    valueA = tensor([1, 2, 4, 1, 3], dtype, device)
    A = SparseTensor(row=rowA, col=colA, value=valueA)

    rowB = paddle.to_tensor([0, 0, 1, 2, 2])
    colB = paddle.to_tensor([1, 2, 2, 1, 2])
    valueB = tensor([2, 3, 1, 2, 4], dtype, device)
    B = SparseTensor(row=rowB, col=colB, value=valueB)

    C = A * B
    rowC, colC, valueC = C.coo()

    assert rowC.tolist() == [0, 2]
    assert colC.tolist() == [2, 1]
    assert valueC.tolist() == [6, 6]

    def jit_mul(A: SparseTensor, B: SparseTensor) -> SparseTensor:
        return mul(A, B)

    jit_mul(A, B)


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_sparse_sparse_mul_empty(dtype, device):
    device = str(device)[6:-1]
    paddle.device.set_device(device)

    rowA = paddle.to_tensor([0])
    colA = paddle.to_tensor([1])
    valueA = tensor([1], dtype, device)
    A = SparseTensor(row=rowA, col=colA, value=valueA)

    rowB = paddle.to_tensor([1])
    colB = paddle.to_tensor([0])
    valueB = tensor([2], dtype, device)
    B = SparseTensor(row=rowB, col=colB, value=valueB)

    C = A * B
    rowC, colC, valueC = C.coo()

    assert rowC.tolist() == []
    assert colC.tolist() == []
    assert valueC.tolist() == []
