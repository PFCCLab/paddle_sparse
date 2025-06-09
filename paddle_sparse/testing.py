from __future__ import annotations

from typing import Any

import paddle

reductions = ["sum", "add", "mean", "min", "max"]

# TODO(beinggod): Disable paddle.float16 because there are some kerenl not support.
dtypes = [paddle.float32, paddle.float64, paddle.int32, paddle.int64]
grad_dtypes = [paddle.float32, paddle.float64]

# TODO(beinggod): Add version on `paddle_scatter` to support bfloat16 testing
# if version.parse(paddle_scatter.__version__) > version.parse("2.0.9"):
#     dtypes.append(paddle.bfloat16)
#     grad_dtypes.append(paddle.bfloat16)

devices = [paddle.CPUPlace()]
if paddle.device.cuda.device_count() > 0:
    devices += [paddle.CUDAPlace(0)]


def tensor(x: Any, dtype: paddle.dtype, device: paddle.base.libpaddle.Place):
    return None if x is None else paddle.to_tensor(x, dtype=dtype, place=device)
