import paddle
import pytest

from paddle_sparse.tensor import SparseTensor
from paddle_sparse.testing import devices
from paddle_sparse.testing import tensor


@pytest.mark.parametrize("device", devices)
def test_permute(device):
    device = str(device)[6:-1]
    paddle.device.set_device(device)

    row, col = tensor([[0, 0, 1, 2, 2], [0, 1, 0, 1, 2]], paddle.int64, device)
    value = tensor([1, 2, 3, 4, 5], paddle.float32, device)
    adj = SparseTensor(row=row, col=col, value=value)

    row, col, value = adj.permute(paddle.to_tensor([1, 0, 2])).coo()
    assert row.tolist() == [0, 1, 1, 2, 2]
    assert col.tolist() == [1, 0, 1, 0, 2]
    assert value.tolist() == [3, 2, 1, 4, 5]
