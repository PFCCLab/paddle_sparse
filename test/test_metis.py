from itertools import product

import pytest
import paddle

from paddle_sparse import SparseTensor
from paddle_sparse.metis import partition, weight2metis
from paddle_sparse.testing import devices, dtypes, tensor, set_testing_device, maybe_skip_testing


def skip_if_cuda_incompatible(dtype, device):
    """Skip test if CUDA device is not compatible with PaddlePaddle."""
    maybe_skip_testing(dtype, device)
    
    # Additional CUDA compatibility check
    device_str = str(device)
    if "gpu" in device_str.lower() or "cuda" in device_str.lower():
        try:
            # Try to set device and create a simple tensor
            original_device = paddle.device.get_device()
            paddle.device.set_device(device_str)
            # Try a simple operation that would fail with incompatible CUDA
            test_tensor = paddle.to_tensor([1.0], dtype='float32')
            _ = test_tensor + 1
            # Restore original device
            if original_device:
                paddle.device.set_device(original_device)
        except Exception:
            pytest.skip(f"CUDA device {device_str} not compatible with PaddlePaddle")


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_partition_single_part(dtype, device):
    """Test partition with single part."""
    skip_if_cuda_incompatible(dtype, device)
    set_testing_device(device)
    
    # Create a simple graph
    row = paddle.to_tensor([0, 1, 2, 3], dtype='int64')
    col = paddle.to_tensor([1, 2, 3, 0], dtype='int64')
    value = tensor([1, 1, 1, 1], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))
    
    partitioned, partptr, perm = partition(sparse_tensor, 1)
    
    # Check partition pointer
    expected_partptr = [0, 4]
    assert partptr.tolist() == expected_partptr, f"Expected {expected_partptr}, got {partptr.tolist()}"
    
    # Check permutation
    assert len(perm) == 4, f"Expected permutation length 4, got {len(perm)}"


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_partition_multiple_parts(dtype, device):
    """Test partition with multiple parts."""
    skip_if_cuda_incompatible(dtype, device)
    set_testing_device(device)
    
    # Create a simple graph
    row = paddle.to_tensor([0, 1, 2, 3], dtype='int64')
    col = paddle.to_tensor([1, 2, 3, 0], dtype='int64')
    value = tensor([1, 1, 1, 1], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))
    
    partitioned, partptr, perm = partition(sparse_tensor, 2)
    
    # Check partition pointer length
    assert len(partptr) == 3, f"Expected partptr length 3, got {len(partptr)}"  # num_parts + 1
    
    # Check that partitions cover all nodes
    total_nodes = partptr[-1].item()
    assert total_nodes == 4, f"Expected total nodes 4, got {total_nodes}"
    
    # Check permutation
    assert len(perm) == 4, f"Expected permutation length 4, got {len(perm)}"


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_partition_with_node_weights(dtype, device):
    """Test partition with node weights."""
    skip_if_cuda_incompatible(dtype, device)
    set_testing_device(device)
    
    # Create a simple graph
    row = paddle.to_tensor([0, 1, 2, 3], dtype='int64')
    col = paddle.to_tensor([1, 2, 3, 0], dtype='int64')
    value = tensor([1, 1, 1, 1], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))
    
    # Create node weights
    node_weight = tensor([1, 2, 3, 4], dtype, device)
    
    partitioned, partptr, perm = partition(sparse_tensor, 2, node_weight=node_weight)
    
    # Check basic properties
    assert len(partptr) == 3, f"Expected partptr length 3, got {len(partptr)}"
    assert len(perm) == 4, f"Expected permutation length 4, got {len(perm)}"


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_partition_balance_edge(dtype, device):
    """Test partition with edge balancing."""
    skip_if_cuda_incompatible(dtype, device)
    set_testing_device(device)
    
    # Create a simple graph
    row = paddle.to_tensor([0, 1, 2, 3], dtype='int64')
    col = paddle.to_tensor([1, 2, 3, 0], dtype='int64')
    value = tensor([1, 1, 1, 1], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))
    
    partitioned, partptr, perm = partition(sparse_tensor, 2, balance_edge=True)
    
    # Check basic properties
    assert len(partptr) == 3, f"Expected partptr length 3, got {len(partptr)}"
    assert len(perm) == 4, f"Expected permutation length 4, got {len(perm)}"


def test_weight2metis():
    """Test weight2metis function."""
    # Test with uniform weights
    uniform_weights = paddle.to_tensor([1.0, 1.0, 1.0], dtype='float32')
    result = weight2metis(uniform_weights)
    assert result is None, "Uniform weights should return None"
    
    # Test with varying weights
    varying_weights = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
    result = weight2metis(varying_weights)
    assert result is not None, "Varying weights should return a tensor"
    assert len(result) == 3, f"Expected length 3, got {len(result)}"


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_sparse_tensor_partition_method(dtype, device):
    """Test SparseTensor.partition method."""
    skip_if_cuda_incompatible(dtype, device)
    set_testing_device(device)
    
    # Create a simple graph
    row = paddle.to_tensor([0, 1, 2, 3], dtype='int64')
    col = paddle.to_tensor([1, 2, 3, 0], dtype='int64')
    value = tensor([1, 1, 1, 1], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))
    
    # Test method call
    partitioned, partptr, perm = sparse_tensor.partition(2)
    
    # Check basic properties
    assert len(partptr) == 3, f"Expected partptr length 3, got {len(partptr)}"
    assert len(perm) == 4, f"Expected permutation length 4, got {len(perm)}"