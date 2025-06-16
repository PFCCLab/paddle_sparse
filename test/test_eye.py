from itertools import product

import numpy as np
import paddle
import pytest

from paddle_sparse.tensor import SparseTensor
from paddle_sparse.testing import devices
from paddle_sparse.testing import dtypes


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_eye(dtype, device):
    device = str(device)[6:-1]
    if device == "cpu" and dtype in [paddle.float16, paddle.bfloat16]:
        pytest.skip(
            reason="Paddle gather_nd CPU kernel not support float16 and bfloat16 dtype."
        )

    paddle.device.set_device(device)

    mat = SparseTensor.eye(3, dtype=dtype, device=device)
    assert str(mat.device())[6:-1] == device
    assert mat.storage.sparse_sizes() == (3, 3)
    assert mat.storage.row().tolist() == [0, 1, 2]
    assert mat.storage.rowptr().tolist() == [0, 1, 2, 3]
    assert mat.storage.col().tolist() == [0, 1, 2]
    # NOTE(beinggod): paddle.Tensor.tolist will interpret bf16 tensor as uint16. We should construct a paddle.Tensor to workaround it.
    np.testing.assert_array_equal(
        mat.storage.value().numpy(),
        paddle.to_tensor([1, 1, 1], dtype=dtype, place=device).numpy(),
    )
    assert mat.storage.value().dtype == dtype
    assert mat.storage.num_cached_keys() == 0

    mat = SparseTensor.eye(3, has_value=False, device=device)
    assert str(mat.device())[6:-1] == str(device)
    assert mat.storage.sparse_sizes() == (3, 3)
    assert mat.storage.row().tolist() == [0, 1, 2]
    assert mat.storage.rowptr().tolist() == [0, 1, 2, 3]
    assert mat.storage.col().tolist() == [0, 1, 2]
    assert mat.storage.value() is None
    assert mat.storage.num_cached_keys() == 0

    mat = SparseTensor.eye(3, 4, fill_cache=True, device=device)
    assert str(mat.device())[6:-1] == str(device)
    assert mat.storage.sparse_sizes() == (3, 4)
    assert mat.storage.row().tolist() == [0, 1, 2]
    assert mat.storage.rowptr().tolist() == [0, 1, 2, 3]
    assert mat.storage.col().tolist() == [0, 1, 2]
    assert mat.storage.num_cached_keys() == 5
    assert mat.storage.rowcount().tolist() == [1, 1, 1]
    assert mat.storage.colptr().tolist() == [0, 1, 2, 3, 3]
    assert mat.storage.colcount().tolist() == [1, 1, 1, 0]
    assert mat.storage.csr2csc().tolist() == [0, 1, 2]
    assert mat.storage.csc2csr().tolist() == [0, 1, 2]

    mat = SparseTensor.eye(4, 3, fill_cache=True, device=device)
    assert str(mat.device())[6:-1] == str(device)
    assert mat.storage.sparse_sizes() == (4, 3)
    assert mat.storage.row().tolist() == [0, 1, 2]
    assert mat.storage.rowptr().tolist() == [0, 1, 2, 3, 3]
    assert mat.storage.col().tolist() == [0, 1, 2]
    assert mat.storage.num_cached_keys() == 5
    assert mat.storage.rowcount().tolist() == [1, 1, 1, 0]
    assert mat.storage.colptr().tolist() == [0, 1, 2, 3]
    assert mat.storage.colcount().tolist() == [1, 1, 1]
    assert mat.storage.csr2csc().tolist() == [0, 1, 2]
    assert mat.storage.csc2csr().tolist() == [0, 1, 2]
