from itertools import product

import numpy as np
import paddle
import pytest

from paddle_sparse import SparseTensor
from paddle_sparse.testing import devices
from paddle_sparse.testing import dtypes
from paddle_sparse.testing import maybe_skip_testing
from paddle_sparse.testing import set_testing_device
from paddle_sparse.testing import tensor


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_add(dtype, device):
    maybe_skip_testing(dtype, device)
    set_testing_device(device)

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
    # NOTE(beinggod): paddle.Tensor.tolist will interpret bf16 tensor as uint16. We should construct a paddle.Tensor to workaround it.
    np.testing.assert_array_equal(
        valueC.numpy(),
        paddle.to_tensor([1, 2, 5, 4, 1, 1, 5, 4], dtype=dtype, place=device).numpy(),
    )
