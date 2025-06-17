from itertools import product

import numpy as np
import paddle
import pytest

from paddle_sparse import SparseTensor
from paddle_sparse.testing import devices
from paddle_sparse.testing import dtypes
from paddle_sparse.testing import tensor


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_sparse_sparse_mul(dtype, device):
    device = str(device)[6:-1]
    if device == "cpu" and dtype in [paddle.float16, paddle.bfloat16]:
        pytest.skip(
            reason="Paddle gather_nd CPU kernel not support float16 and bfloat16 dtype."
        )

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
    # NOTE(beinggod): paddle.Tensor.tolist will interpret bf16 tensor as uint16. We should construct a paddle.Tensor to workaround it.
    np.testing.assert_array_equal(
        valueC.numpy(), paddle.to_tensor([6, 6], dtype=dtype, place=device).numpy()
    )


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_sparse_sparse_mul_empty(dtype, device):
    device = str(device)[6:-1]
    if device == "cpu" and dtype in [paddle.float16, paddle.bfloat16]:
        pytest.skip(
            reason="Paddle gather_nd CPU kernel not support float16 and bfloat16 dtype."
        )

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
