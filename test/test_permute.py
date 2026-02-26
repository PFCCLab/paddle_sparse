import paddle
import pytest

from paddle_sparse.tensor import SparseTensor
from paddle_sparse.testing import devices
from paddle_sparse.testing import set_testing_device
from paddle_sparse.testing import tensor


def skip_if_cuda_incompatible(device):
    """Skip test if CUDA device is not compatible with PaddlePaddle."""
    device_str = str(device)
    if "gpu" in device_str.lower() or "cuda" in device_str.lower():
        try:
            original_device = paddle.device.get_device()
            paddle.device.set_device(device_str)
            test_tensor = paddle.to_tensor([1.0], dtype='float32')
            _ = test_tensor + 1
            if original_device:
                paddle.device.set_device(original_device)
        except Exception:
            pytest.skip(f"CUDA device {device_str} not compatible with PaddlePaddle")


@pytest.mark.parametrize("device", devices)
def test_permute(device):
    skip_if_cuda_incompatible(device)
    set_testing_device(device)

    row, col = tensor([[0, 0, 1, 2, 2], [0, 1, 0, 1, 2]], paddle.int64, device)
    value = tensor([1, 2, 3, 4, 5], paddle.float32, device)
    adj = SparseTensor(row=row, col=col, value=value)

    row, col, value = adj.permute(paddle.to_tensor([1, 0, 2])).coo()
    assert row.tolist() == [0, 1, 1, 2, 2]
    assert col.tolist() == [1, 0, 1, 0, 2]
    assert value.tolist() == [3, 2, 1, 4, 5]
