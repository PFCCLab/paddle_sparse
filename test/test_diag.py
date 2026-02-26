from itertools import product

import pytest
import paddle
import numpy as np

from paddle_sparse import SparseTensor
from paddle_sparse.diag import remove_diag, set_diag, fill_diag, get_diag
from paddle_sparse.testing import devices, dtypes, tensor, set_testing_device


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_get_diag(dtype, device):
    """Test get_diag functionality."""
    set_testing_device(device)
    
    # Create a sparse matrix with diagonal elements
    row = paddle.to_tensor([0, 0, 1, 1, 2], dtype='int64')
    col = paddle.to_tensor([0, 1, 1, 2, 2], dtype='int64')
    value = tensor([1, 2, 3, 4, 5], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    diag = get_diag(sparse_tensor)
    
    # Check diagonal values using numpy to avoid bfloat16.tolist() bug
    # For bfloat16, we need to cast to float32 first before converting to numpy
    if diag.dtype == paddle.bfloat16:
        diag_float32 = diag.cast('float32')
        actual_diag = diag_float32.numpy()
    else:
        actual_diag = diag.numpy()
    expected_diag = np.array([1, 3, 5], dtype='float32')
    np.testing.assert_array_almost_equal(actual_diag, expected_diag, decimal=5)


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_remove_diag(dtype, device):
    """Test remove_diag functionality."""
    set_testing_device(device)
    
    # Create a sparse matrix with diagonal elements
    row = paddle.to_tensor([0, 0, 1, 1, 2], dtype='int64')
    col = paddle.to_tensor([0, 1, 1, 2, 2], dtype='int64')
    value = tensor([1, 2, 3, 4, 5], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    no_diag = remove_diag(sparse_tensor)
    row_no_diag, col_no_diag, value_no_diag = no_diag.coo()
    
    # Should have removed diagonal elements (0,0), (1,1), (2,2)
    expected_nnz = 2  # Only (0,1) and (1,2) remain
    assert len(row_no_diag) == expected_nnz, f"Expected {expected_nnz} non-zeros, got {len(row_no_diag)}"


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_set_diag(dtype, device):
    """Test set_diag functionality."""
    set_testing_device(device)
    
    # Create a sparse matrix without diagonal elements
    row = paddle.to_tensor([0, 1], dtype='int64')
    col = paddle.to_tensor([1, 2], dtype='int64')
    value = tensor([2, 4], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    # Set diagonal values
    diag_values = tensor([10, 20, 30], dtype, device)
    with_diag = set_diag(sparse_tensor, diag_values)
    
    # Check that diagonal was added
    row_with_diag, col_with_diag, value_with_diag = with_diag.coo()
    assert len(row_with_diag) >= 5, "Should have original edges plus diagonal elements"


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_fill_diag(dtype, device):
    """Test fill_diag functionality."""
    set_testing_device(device)
    
    # Create a sparse matrix
    row = paddle.to_tensor([0, 1], dtype='int64')
    col = paddle.to_tensor([1, 2], dtype='int64')
    value = tensor([2, 4], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    # Fill diagonal with a specific value
    fill_value = 99.0
    filled = fill_diag(sparse_tensor, fill_value)
    
    # Check that diagonal was filled
    row_filled, col_filled, value_filled = filled.coo()
    assert len(row_filled) >= 5, "Should have original edges plus diagonal elements"


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_sparse_tensor_diag_methods(dtype, device):
    """Test SparseTensor diagonal methods."""
    set_testing_device(device)
    
    # Create a sparse matrix
    row = paddle.to_tensor([0, 0, 1, 2], dtype='int64')
    col = paddle.to_tensor([0, 1, 1, 2], dtype='int64')
    value = tensor([1, 2, 3, 4], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    # Test get_diag method
    diag = sparse_tensor.get_diag()
    # For bfloat16, we need to cast to float32 first before converting to numpy
    if diag.dtype == paddle.bfloat16:
        diag_float32 = diag.cast('float32')
        actual_diag = diag_float32.numpy()
    else:
        actual_diag = diag.numpy()
    expected_diag = np.array([1, 3, 4], dtype='float32')
    np.testing.assert_array_almost_equal(actual_diag, expected_diag, decimal=5)
    
    # Test remove_diag method
    no_diag = sparse_tensor.remove_diag()
    row_no_diag, _, _ = no_diag.coo()
    assert len(row_no_diag) == 1, "Should have only one off-diagonal element"
    
    # Test fill_diag method
    filled = sparse_tensor.fill_diag(5.0)
    row_filled, _, _ = filled.coo()
    assert len(row_filled) >= 3, "Should have at least diagonal elements"