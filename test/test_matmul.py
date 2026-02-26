from itertools import product

import pytest
import paddle

from paddle_sparse import SparseTensor, matmul
from paddle_sparse.testing import devices, dtypes, tensor, set_testing_device


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_matmul_dense(dtype, device):
    """Test sparse-dense matrix multiplication."""
    set_testing_device(device)
    
    # Skip float16 and bfloat16 due to kernel not supported
    if dtype in [paddle.float16, paddle.bfloat16]:
        pytest.skip()
    
    # Create sparse tensor
    row = paddle.to_tensor([0, 0, 1, 2, 2], dtype='int64')
    col = paddle.to_tensor([0, 2, 1, 0, 1], dtype='int64')
    value = tensor([1, 2, 4, 1, 3], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    # Create dense tensor
    dense_tensor = tensor([[1, 4], [2, 5], [3, 6]], dtype, device)
    
    # Test matmul
    out = matmul(sparse_tensor, dense_tensor)
    expected = [[7, 16], [8, 20], [7, 19]]
    
    out_list = out.tolist()
    assert out_list == expected, f"Expected {expected}, got {out_list}"


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_matmul_reduce_operations(dtype, device):
    """Test different reduce operations for sparse-dense matmul."""
    set_testing_device(device)
    
    # Skip float16 and bfloat16 due to kernel not supported
    if dtype in [paddle.float16, paddle.bfloat16]:
        pytest.skip()
    
    # Create sparse tensor with multiple values per row
    row = paddle.to_tensor([0, 0, 1, 1, 2], dtype='int64')
    col = paddle.to_tensor([0, 1, 0, 1, 1], dtype='int64')
    value = tensor([1, 2, 3, 4, 5], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 2))
    
    # Create dense tensor
    dense_tensor = tensor([[1], [2]], dtype, device)
    
    # Test sum reduction (default)
    out_sum = matmul(sparse_tensor, dense_tensor, reduce="sum")
    
    # Test mean reduction
    out_mean = matmul(sparse_tensor, dense_tensor, reduce="mean")
    
    # Test min reduction
    out_min = matmul(sparse_tensor, dense_tensor, reduce="min")
    
    # Test max reduction
    out_max = matmul(sparse_tensor, dense_tensor, reduce="max")
    
    # Basic shape checks
    assert out_sum.shape == (3, 1)
    assert out_mean.shape == (3, 1)
    assert out_min.shape == (3, 1)
    assert out_max.shape == (3, 1)


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_sparse_tensor_matmul_method(dtype, device):
    """Test SparseTensor.matmul method."""
    set_testing_device(device)
    
    # Skip float16 and bfloat16 due to kernel not supported
    if dtype in [paddle.float16, paddle.bfloat16]:
        pytest.skip()
    
    # Create sparse tensor
    row = paddle.to_tensor([0, 1, 2], dtype='int64')
    col = paddle.to_tensor([0, 1, 2], dtype='int64')
    value = tensor([1, 2, 3], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    # Create dense tensor
    dense_tensor = tensor([[1], [2], [3]], dtype, device)
    
    # Test method call
    out = sparse_tensor.matmul(dense_tensor)
    expected = [[1], [4], [9]]
    
    out_list = out.tolist()
    assert out_list == expected, f"Expected {expected}, got {out_list}"


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_sparse_tensor_matmul_operator(dtype, device):
    """Test SparseTensor @ operator."""
    set_testing_device(device)
    
    # Skip float16 and bfloat16 due to kernel not supported
    if dtype in [paddle.float16, paddle.bfloat16]:
        pytest.skip()
    
    # Create sparse tensor
    row = paddle.to_tensor([0, 1, 2], dtype='int64')
    col = paddle.to_tensor([0, 1, 2], dtype='int64')
    value = tensor([1, 2, 3], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    # Create dense tensor
    dense_tensor = tensor([[1], [2], [3]], dtype, device)
    
    # Test @ operator
    out = sparse_tensor @ dense_tensor
    expected = [[1], [4], [9]]
    
    out_list = out.tolist()
    assert out_list == expected, f"Expected {expected}, got {out_list}"