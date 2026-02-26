from itertools import product

import pytest
import paddle

from paddle_sparse import SparseTensor
from paddle_sparse.rw import random_walk
from paddle_sparse.testing import devices, dtypes, tensor, set_testing_device


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_random_walk(dtype, device):
    """Test random walk functionality."""
    set_testing_device(device)
    
    # Create a simple graph: 0 -> 1 -> 2 -> 0
    row = paddle.to_tensor([0, 1, 2], dtype='int64')
    col = paddle.to_tensor([1, 2, 0], dtype='int64')
    value = tensor([1, 1, 1], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    # Start random walks from nodes 0 and 1
    start = paddle.to_tensor([0, 1], dtype='int64')
    walk_length = 3
    
    walks = random_walk(sparse_tensor, start, walk_length)
    
    # Check output shape
    expected_shape = [2, 4]  # batch_size=2, walk_length+1=4
    assert list(walks.shape) == expected_shape, f"Expected shape {expected_shape}, got {list(walks.shape)}"
    
    # Check that walks start with the correct nodes
    assert walks[0, 0].item() == 0, "First walk should start with node 0"
    assert walks[1, 0].item() == 1, "Second walk should start with node 1"


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_sparse_tensor_random_walk_method(dtype, device):
    """Test SparseTensor.random_walk method."""
    set_testing_device(device)
    
    # Create a simple graph
    row = paddle.to_tensor([0, 1, 2], dtype='int64')
    col = paddle.to_tensor([1, 2, 0], dtype='int64')
    value = tensor([1, 1, 1], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    # Test method call
    start = paddle.to_tensor([0], dtype='int64')
    walks = sparse_tensor.random_walk(start, 2)
    
    # Check output shape
    expected_shape = [1, 3]  # batch_size=1, walk_length+1=3
    assert list(walks.shape) == expected_shape, f"Expected shape {expected_shape}, got {list(walks.shape)}"