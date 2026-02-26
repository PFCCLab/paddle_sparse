import paddle
import pytest

from paddle_sparse import coalesce


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


def test_coalesce():
    skip_if_cuda_incompatible()
    paddle.device.set_device('cpu')
    row = paddle.to_tensor([1, 0, 1, 0, 2, 1])
    col = paddle.to_tensor([0, 1, 1, 1, 0, 0])
    index = paddle.stack([row, col], axis=0)

    index, _ = coalesce(index, None, m=3, n=2)
    assert index.tolist() == [[0, 1, 1, 2], [1, 0, 1, 0]]


def test_coalesce_add():
    skip_if_cuda_incompatible()
    paddle.device.set_device('cpu')
    row = paddle.to_tensor([1, 0, 1, 0, 2, 1])
    col = paddle.to_tensor([0, 1, 1, 1, 0, 0])
    index = paddle.stack([row, col], axis=0)
    value = paddle.to_tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])

    index, value = coalesce(index, value, m=3, n=2)
    assert index.tolist() == [[0, 1, 1, 2], [1, 0, 1, 0]]
    assert value.tolist() == [[6, 8], [7, 9], [3, 4], [5, 6]]


def test_coalesce_max():
    skip_if_cuda_incompatible()
    paddle.device.set_device('cpu')
    row = paddle.to_tensor([1, 0, 1, 0, 2, 1])
    col = paddle.to_tensor([0, 1, 1, 1, 0, 0])
    index = paddle.stack([row, col], axis=0)
    value = paddle.to_tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])

    index, value = coalesce(index, value, m=3, n=2, op="max")
    assert index.tolist() == [[0, 1, 1, 2], [1, 0, 1, 0]]
    assert value.tolist() == [[4, 5], [6, 7], [3, 4], [5, 6]]
