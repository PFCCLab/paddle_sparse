import paddle
import pytest

from paddle_sparse.tensor import SparseTensor


def skip_if_cuda_incompatible():
    """Skip test if CUDA device is not compatible with PaddlePaddle."""
    if paddle.device.cuda.device_count() > 0:
        try:
            device = paddle.CUDAPlace(0)
            original_device = paddle.device.get_device()
            paddle.device.set_device(device)
            test_tensor = paddle.to_tensor([1.0], dtype='float32')
            _ = test_tensor + 1
            if original_device:
                paddle.device.set_device(original_device)
        except Exception:
            pytest.skip("CUDA device not compatible with PaddlePaddle")


def test_overload():
    skip_if_cuda_incompatible()
    paddle.device.set_device('cpu')
    row = paddle.to_tensor([0, 1, 1, 2, 2])
    col = paddle.to_tensor([1, 0, 2, 1, 2])
    mat = SparseTensor(row=row, col=col)

    other = paddle.to_tensor([1, 2, 3]).view([3, 1])
    other + mat
    mat + other
    other * mat
    mat * other

    other = paddle.to_tensor([1, 2, 3]).view([1, 3])
    other + mat
    mat + other
    other * mat
    mat * other
