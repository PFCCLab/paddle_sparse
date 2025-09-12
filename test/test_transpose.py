from itertools import product

import numpy as np
import paddle
import pytest

from paddle_sparse import transpose
from paddle_sparse.testing import devices
from paddle_sparse.testing import dtypes
from paddle_sparse.testing import maybe_skip_testing
from paddle_sparse.testing import set_testing_device
from paddle_sparse.testing import tensor


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_transpose_matrix(dtype, device):
    maybe_skip_testing(dtype, device)
    set_testing_device(device)

    row = paddle.to_tensor([1, 0, 1, 2])
    col = paddle.to_tensor([0, 1, 1, 0])
    index = paddle.stack([row, col], axis=0)
    value = tensor([1, 2, 3, 4], dtype, device)

    index, value = transpose(index, value, m=3, n=2)
    assert index.tolist() == [[0, 0, 1, 1], [1, 2, 0, 1]]
    # NOTE(beinggod): paddle.Tensor.tolist will interpret bf16 tensor as uint16. We should construct a paddle.Tensor to workaround it.
    np.testing.assert_array_equal(
        value.numpy(), paddle.to_tensor([1, 4, 2, 3], dtype=dtype, place=device).numpy()
    )


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_transpose(dtype, device):
    maybe_skip_testing(dtype, device)
    set_testing_device(device)

    row = paddle.to_tensor([1, 0, 1, 0, 2, 1])
    col = paddle.to_tensor([0, 1, 1, 1, 0, 0])
    index = paddle.stack([row, col], axis=0)
    value = tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]], dtype, device)

    index, value = transpose(index, value, m=3, n=2)
    assert index.tolist() == [[0, 0, 1, 1], [1, 2, 0, 1]]
    # NOTE(beinggod): paddle.Tensor.tolist will interpret bf16 tensor as uint16. We should construct a paddle.Tensor to workaround it.
    np.testing.assert_array_equal(
        value.numpy(),
        paddle.to_tensor(
            [[7, 9], [5, 6], [6, 8], [3, 4]], dtype=dtype, place=device
        ).numpy(),
    )
