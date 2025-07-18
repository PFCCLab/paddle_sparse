from itertools import product

import numpy as np
import paddle
import paddle_sparse_ops
import pytest

from paddle_sparse.storage import SparseStorage
from paddle_sparse.testing import devices
from paddle_sparse.testing import dtypes
from paddle_sparse.testing import tensor


@pytest.mark.parametrize("device", devices)
def test_ind2ptr(device):
    device = str(device)[6:-1]
    paddle.device.set_device(device)

    row = tensor([2, 2, 4, 5, 5, 6], paddle.int64, device)
    rowptr = paddle_sparse_ops.ind2ptr(row, 8)
    assert rowptr.tolist() == [0, 0, 0, 2, 2, 3, 5, 6, 6]

    row = paddle_sparse_ops.ptr2ind(rowptr, 6)
    assert row.tolist() == [2, 2, 4, 5, 5, 6]

    row = tensor([], paddle.int64, device)
    rowptr = paddle_sparse_ops.ind2ptr(row, 8)
    assert rowptr.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0]

    row = paddle_sparse_ops.ptr2ind(rowptr, 0)
    assert row.tolist() == []


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_storage(dtype, device):
    device = str(device)[6:-1]
    if device == "cpu" and dtype in [paddle.float16, paddle.bfloat16]:
        pytest.skip(
            reason="Paddle gather_nd CPU kernel not support float16 and bfloat16 dtype."
        )

    paddle.device.set_device(device)

    row, col = tensor([[0, 0, 1, 1], [0, 1, 0, 1]], paddle.int64, device)

    storage = SparseStorage(row=row, col=col)
    assert storage.row().tolist() == [0, 0, 1, 1]
    assert storage.col().tolist() == [0, 1, 0, 1]
    assert storage.value() is None
    assert storage.sparse_sizes() == (2, 2)

    row, col = tensor([[0, 0, 1, 1], [1, 0, 1, 0]], paddle.int64, device)
    value = tensor([2, 1, 4, 3], dtype, device)
    storage = SparseStorage(row=row, col=col, value=value)
    assert storage.row().tolist() == [0, 0, 1, 1]
    assert storage.col().tolist() == [0, 1, 0, 1]
    # NOTE(beinggod): paddle.Tensor.tolist will interpret bf16 tensor as uint16. We should construct a paddle.Tensor to workaround it.
    np.testing.assert_array_equal(
        storage.value().numpy(),
        paddle.to_tensor([1, 2, 3, 4], dtype=dtype, place=device).numpy(),
    )
    assert storage.sparse_sizes() == (2, 2)


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_caching(dtype, device):
    device = str(device)[6:-1]
    paddle.device.set_device(device)

    row, col = tensor([[0, 0, 1, 1], [0, 1, 0, 1]], paddle.int64, device)
    storage = SparseStorage(row=row, col=col)

    assert storage._row.tolist() == row.tolist()
    assert storage._col.tolist() == col.tolist()
    assert storage._value is None

    assert storage._rowcount is None
    assert storage._rowptr is None
    assert storage._colcount is None
    assert storage._colptr is None
    assert storage._csr2csc is None
    assert storage.num_cached_keys() == 0

    storage.fill_cache_()
    assert storage._rowcount.tolist() == [2, 2]
    assert storage._rowptr.tolist() == [0, 2, 4]
    assert storage._colcount.tolist() == [2, 2]
    assert storage._colptr.tolist() == [0, 2, 4]
    assert storage._csr2csc.tolist() == [0, 2, 1, 3]
    assert storage._csc2csr.tolist() == [0, 2, 1, 3]
    assert storage.num_cached_keys() == 5

    storage = SparseStorage(
        row=row,
        rowptr=storage._rowptr,
        col=col,
        value=storage._value,
        sparse_sizes=storage._sparse_sizes,
        rowcount=storage._rowcount,
        colptr=storage._colptr,
        colcount=storage._colcount,
        csr2csc=storage._csr2csc,
        csc2csr=storage._csc2csr,
    )

    assert storage._rowcount.tolist() == [2, 2]
    assert storage._rowptr.tolist() == [0, 2, 4]
    assert storage._colcount.tolist() == [2, 2]
    assert storage._colptr.tolist() == [0, 2, 4]
    assert storage._csr2csc.tolist() == [0, 2, 1, 3]
    assert storage._csc2csr.tolist() == [0, 2, 1, 3]
    assert storage.num_cached_keys() == 5

    storage.clear_cache_()
    assert storage._rowcount is None
    assert storage._rowptr is not None
    assert storage._colcount is None
    assert storage._colptr is None
    assert storage._csr2csc is None
    assert storage.num_cached_keys() == 0


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_utility(dtype, device):
    device = str(device)[6:-1]
    if device == "cpu" and dtype in [paddle.float16, paddle.bfloat16]:
        pytest.skip(
            reason="Paddle gather_nd CPU kernel not support float16 and bfloat16 dtype."
        )

    paddle.device.set_device(device)

    row, col = tensor([[0, 0, 1, 1], [1, 0, 1, 0]], paddle.int64, device)
    value = tensor([1, 2, 3, 4], dtype, device)
    storage = SparseStorage(row=row, col=col, value=value)

    assert storage.has_value()

    storage.set_value_(value, layout="csc")
    # NOTE(beinggod): paddle.Tensor.tolist will interpret bf16 tensor as uint16. We should construct a paddle.Tensor to workaround it.
    np.testing.assert_array_equal(
        storage.value().numpy(),
        paddle.to_tensor([1, 3, 2, 4], dtype=dtype, place=device).numpy(),
    )
    storage.set_value_(value, layout="coo")
    np.testing.assert_array_equal(
        storage.value().numpy(),
        paddle.to_tensor([1, 2, 3, 4], dtype=dtype, place=device).numpy(),
    )

    storage = storage.set_value(value, layout="csc")
    np.testing.assert_array_equal(
        storage.value().numpy(),
        paddle.to_tensor([1, 3, 2, 4], dtype=dtype, place=device).numpy(),
    )
    storage = storage.set_value(value, layout="coo")
    np.testing.assert_array_equal(
        storage.value().numpy(),
        paddle.to_tensor([1, 2, 3, 4], dtype=dtype, place=device).numpy(),
    )

    storage = storage.sparse_resize((3, 3))
    assert storage.sparse_sizes() == (3, 3)

    new_storage = storage.copy()
    assert new_storage != storage
    assert new_storage.col().data_ptr() == storage.col().data_ptr()

    new_storage = storage.clone()
    assert new_storage != storage
    assert new_storage.col().data_ptr() != storage.col().data_ptr()


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_coalesce(dtype, device):
    device = str(device)[6:-1]
    if device == "cpu" and dtype in [paddle.float16, paddle.bfloat16]:
        pytest.skip(
            reason="Paddle segment_csr_cpu_forward_kernel not support float16 and bfloat16 dtype."
        )

    paddle.device.set_device(device)

    row, col = tensor([[0, 0, 0, 1, 1], [0, 1, 1, 0, 1]], paddle.int64, device)
    value = tensor([1, 1, 1, 3, 4], dtype, device)
    storage = SparseStorage(row=row, col=col, value=value)

    assert storage.row().tolist() == row.tolist()
    assert storage.col().tolist() == col.tolist()
    assert storage.value().tolist() == value.tolist()

    assert not storage.is_coalesced()
    storage = storage.coalesce()
    assert storage.is_coalesced()

    assert storage.row().tolist() == [0, 0, 1, 1]
    assert storage.col().tolist() == [0, 1, 0, 1]
    # NOTE(beinggod): paddle.Tensor.tolist will interpret bf16 tensor as uint16. We should construct a paddle.Tensor to workaround it.
    np.testing.assert_array_equal(
        storage.value().numpy(),
        paddle.to_tensor([1, 2, 3, 4], dtype=dtype, place=device).numpy(),
    )


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_sparse_reshape(dtype, device):
    device = str(device)[6:-1]
    if device == "cpu" and dtype in [paddle.float16, paddle.bfloat16]:
        pytest.skip(
            reason="Paddle segment_csr_cpu_forward_kernel not support float16 and bfloat16 dtype."
        )

    paddle.device.set_device(device)

    row, col = tensor([[0, 1, 2, 3], [0, 1, 2, 3]], paddle.int64, device)
    storage = SparseStorage(row=row, col=col)

    storage = storage.sparse_reshape(2, 8)
    assert storage.sparse_sizes() == (2, 8)
    assert storage.row().tolist() == [0, 0, 1, 1]
    assert storage.col().tolist() == [0, 5, 2, 7]

    storage = storage.sparse_reshape(-1, 4)
    assert storage.sparse_sizes() == (4, 4)
    assert storage.row().tolist() == [0, 1, 2, 3]
    assert storage.col().tolist() == [0, 1, 2, 3]

    storage = storage.sparse_reshape(2, -1)
    assert storage.sparse_sizes() == (2, 8)
    assert storage.row().tolist() == [0, 0, 1, 1]
    assert storage.col().tolist() == [0, 5, 2, 7]
