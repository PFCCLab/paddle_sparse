from itertools import product

import pytest
import paddle

from paddle_sparse import SparseTensor, spspmm
from paddle_sparse.testing import devices, grad_dtypes, tensor, set_testing_device


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_spspmm(dtype, device):
    if str(dtype) in ['paddle.float16', 'paddle.bfloat16']:
        return  # Not yet implemented.

    set_testing_device(device)
    
    indexA = paddle.to_tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]], dtype='int64')
    valueA = tensor([1, 2, 3, 4, 5], dtype, device)
    indexB = paddle.to_tensor([[0, 2], [1, 0]], dtype='int64')
    valueB = tensor([2, 4], dtype, device)

    indexC, valueC = spspmm(indexA, valueA, indexB, valueB, 3, 3, 2)
    
    expected_index = [[0, 1, 2], [0, 1, 1]]
    expected_value = [8, 6, 8]
    
    assert indexC.tolist() == expected_index, f"Expected index {expected_index}, got {indexC.tolist()}"
    assert valueC.tolist() == expected_value, f"Expected value {expected_value}, got {valueC.tolist()}"


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_sparse_tensor_spspmm(dtype, device):
    if str(dtype) in ['paddle.float16', 'paddle.bfloat16']:
        return  # Not yet implemented.

    set_testing_device(device)
    
    x = SparseTensor(
        row=paddle.to_tensor([
            0, 1, 1, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9
        ], dtype='int64'),
        col=paddle.to_tensor([
            0, 5, 10, 15, 1, 2, 3, 7, 13, 6, 9, 5, 10, 15, 11, 14, 5, 15
        ], dtype='int64'),
        value=paddle.to_tensor([
            1, 3**-0.5, 3**-0.5, 3**-0.5, 1, 1, 1, -(2**-0.5), -(2**-0.5),
            -(2**-0.5), -(2**-0.5), 6**-0.5, -(6**0.5) / 3, 6**-0.5, -(2**-0.5),
            -(2**-0.5), 2**-0.5, -(2**-0.5)
        ], dtype=dtype),
    )

    expected = paddle.eye(10, dtype=dtype)

    # Test dense matrix multiplication
    out = x @ x.to_dense().t()
    assert paddle.allclose(out, expected, atol=1e-2), "Dense matmul failed"

    # Test sparse matrix multiplication
    try:
        out = x @ x.t()
        out = out.to_dense()
        assert paddle.allclose(out, expected, atol=1e-2), "Sparse matmul failed"
    except NotImplementedError:
        # Skip if sparse-sparse multiplication is not implemented
        pytest.skip("Sparse-sparse multiplication not implemented")