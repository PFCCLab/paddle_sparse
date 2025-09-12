import paddle
import pytest

from paddle_sparse.cat import cat
from paddle_sparse.tensor import SparseTensor
from paddle_sparse.testing import devices
from paddle_sparse.testing import set_testing_device
from paddle_sparse.testing import tensor


@pytest.mark.parametrize("device", devices)
def test_cat(device):
    set_testing_device(device)

    row, col = tensor([[0, 0, 1], [0, 1, 2]], paddle.int64, device)
    mat1 = SparseTensor(row=row, col=col)
    mat1.fill_cache_()

    row, col = tensor([[0, 0, 1, 2], [0, 1, 1, 0]], paddle.int64, device)
    mat2 = SparseTensor(row=row, col=col)
    mat2.fill_cache_()

    out = cat([mat1, mat2], dim=0)
    assert out.to_dense().tolist() == [
        [1, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
    ]
    assert out.storage.has_row()
    assert out.storage.has_rowptr()
    assert out.storage.has_rowcount()
    assert out.storage.num_cached_keys() == 1

    out = cat([mat1, mat2], dim=1)
    assert out.to_dense().tolist() == [
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ]
    assert out.storage.has_row()
    assert not out.storage.has_rowptr()
    assert out.storage.num_cached_keys() == 2

    out = cat([mat1, mat2], dim=(0, 1))
    assert out.to_dense().tolist() == [
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ]
    assert out.storage.has_row()
    assert out.storage.has_rowptr()
    assert out.storage.num_cached_keys() == 5

    value = paddle.randn((mat1.nnz(), 4))
    mat1 = mat1.set_value_(value, layout="coo")
    out = cat([mat1, mat1], dim=-1)
    assert out.storage.value().shape == [mat1.nnz().item(), 8]
    assert out.storage.has_row()
    assert out.storage.has_rowptr()
    assert out.storage.num_cached_keys() == 5
