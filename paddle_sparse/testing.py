from __future__ import annotations

from typing import Any

import paddle
import paddle_scatter
import pytest
from packaging import version

reductions = ["sum", "add", "mean", "min", "max"]

dtypes = [paddle.float16, paddle.float32, paddle.float64, paddle.int32, paddle.int64]
grad_dtypes = [paddle.float32, paddle.float64]

if version.parse(paddle_scatter.__version__) > version.parse("2.0.9"):
    dtypes.append(paddle.bfloat16)
    grad_dtypes.append(paddle.bfloat16)

devices = [paddle.CPUPlace()]
if paddle.device.cuda.device_count() > 0:
    devices += [paddle.CUDAPlace(0)]


def tensor(x: Any, dtype: paddle.dtype, device: paddle.base.libpaddle.Place):
    return None if x is None else paddle.to_tensor(x, dtype=dtype, place=device)


def maybe_skip_testing(dtype: paddle.dtype, device: paddle.base.libpaddle.Place):
    device = str(device)[6:-1]
    if device == "cpu" and dtype in [paddle.float16, paddle.bfloat16]:
        pytest.skip()
    
    # Skip CUDA tests if device has compute capability not supported by PaddlePaddle
    # This handles cases like CUDA error 700 (illegal memory access) or
    # error 209 (no kernel image is available)
    if device.startswith("gpu:") or device.startswith("cuda:"):
        # Try to create a small tensor on the device to test compatibility
        try:
            paddle.device.set_device(device)
            test_tensor = paddle.to_tensor([1.0], dtype='float32', place=device)
            # Force a simple CUDA operation to verify kernel availability
            _ = test_tensor + 1
        except Exception:
            pytest.skip(f"CUDA device {device} not compatible with PaddlePaddle")


def set_testing_device(device: paddle.base.libpaddle.Place):
    device = str(device)[6:-1]
    paddle.device.set_device(device)
