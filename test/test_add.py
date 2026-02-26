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


def skip_if_cuda_incompatible(dtype, device):
    """Skip test if CUDA device is not compatible with PaddlePaddle."""
    maybe_skip_testing(dtype, device)
    
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


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_add(dtype, device):
    skip_if_cuda_incompatible(dtype, device)
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
