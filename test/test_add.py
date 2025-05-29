from itertools import product

import paddle
import pytest

from paddle_sparse import SparseTensor
from paddle_sparse import add
from paddle_sparse.testing import devices
from paddle_sparse.testing import dtypes
from paddle_sparse.testing import tensor


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_add(dtype, device):
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

    C = A + B
    rowC, colC, valueC = C.coo()

    assert rowC.tolist() == [0, 0, 0, 1, 1, 2, 2, 2]
    assert colC.tolist() == [0, 1, 2, 1, 2, 0, 1, 2]
    assert valueC.tolist() == [1, 2, 5, 4, 1, 1, 5, 4]

    def jit_add(A: SparseTensor, B: SparseTensor) -> SparseTensor:
        return add(A, B)

    jit_add(A, B)
