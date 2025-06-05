from itertools import product

import paddle
import pytest

from paddle_sparse import SparseTensor
from paddle_sparse.testing import devices
from paddle_sparse.testing import dtypes


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_reduce_sum(dtype, device):
    device = str(device)[6:-1]
    paddle.device.set_device(device)

    row = paddle.to_tensor([1, 0, 1, 0, 2, 1])
    col = paddle.to_tensor([0, 1, 1, 1, 0, 0])
    value = paddle.to_tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    t = SparseTensor(row=row, col=col, value=value)

    assert t.sum() == value.sum()


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_reduce_mean(dtype, device):
    device = str(device)[6:-1]
    paddle.device.set_device(device)

    row = paddle.to_tensor([1, 0, 1, 0, 2, 1])
    col = paddle.to_tensor([0, 1, 1, 1, 0, 0])
    value = paddle.to_tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    t = SparseTensor(row=row, col=col, value=value)

    assert t.mean() == value.mean()


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_reduce_max(dtype, device):
    device = str(device)[6:-1]
    paddle.device.set_device(device)

    row = paddle.to_tensor([1, 0, 1, 0, 2, 1])
    col = paddle.to_tensor([0, 1, 1, 1, 0, 0])
    value = paddle.to_tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    t = SparseTensor(row=row, col=col, value=value)

    assert t.max() == value.max()


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_reduce_min(dtype, device):
    device = str(device)[6:-1]
    paddle.device.set_device(device)

    row = paddle.to_tensor([1, 0, 1, 0, 2, 1])
    col = paddle.to_tensor([0, 1, 1, 1, 0, 0])
    value = paddle.to_tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    t = SparseTensor(row=row, col=col, value=value)

    assert t.min() == value.min()
