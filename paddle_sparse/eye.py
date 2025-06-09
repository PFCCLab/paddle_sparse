from __future__ import annotations

import paddle


def eye(m, dtype=None, device=None):
    """Returns a sparse matrix with ones on the diagonal and zeros elsewhere.

    Args:
        m (int): The first dimension of sparse matrix.
        dtype (`paddle.dtype`, optional): The desired data type of returned
            value vector. (default is set by `paddle.set_default_tensor_type()`)
        device (`paddle.device`, optional): The desired device of returned
            tensors. (default is set by `paddle.set_default_tensor_type()`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    row = paddle.arange(m, dtype=paddle.int64).to(device)
    index = paddle.stack([row, row], axis=0)

    value = paddle.ones(m, dtype=dtype).to(device)

    return index, value
