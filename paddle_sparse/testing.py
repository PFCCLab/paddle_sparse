from __future__ import annotations

from typing import Any

import paddle
import paddle_scatter
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
