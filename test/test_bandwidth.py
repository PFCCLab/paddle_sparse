from itertools import product

import pytest
import paddle

from paddle_sparse import SparseTensor
from paddle_sparse.bandwidth import reverse_cuthill_mckee
from paddle_sparse.testing import devices, dtypes, tensor, set_testing_device, maybe_skip_testing


def skip_if_unsupported_dtype(dtype, device):
    """Skip test if dtype is not supported by the operation."""
    maybe_skip_testing(dtype, device)
    
    # paddle_scatter's segment_csr does not support float16
    if dtype == paddle.float16:
        pytest.skip("float16 is not supported by paddle_scatter.segment_csr")
    
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
def test_reverse_cuthill_mckee(dtype, device):
    """Test reverse Cuthill-McKee ordering."""
    skip_if_unsupported_dtype(dtype, device)
    set_testing_device(device)
    
    # Create a simple graph
    row = paddle.to_tensor([0, 1, 2, 0, 1], dtype='int64')
    col = paddle.to_tensor([1, 2, 0, 2, 0], dtype='int64')
    value = tensor([1, 1, 1, 1, 1], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    reordered, perm = reverse_cuthill_mckee(sparse_tensor)
    
    # Check that permutation has correct length
    assert len(perm) == 3, f"Expected permutation length 3, got {len(perm)}"
    
    # Check that reordered tensor has same size
    assert reordered.sparse_sizes() == (3, 3), f"Expected size (3, 3), got {reordered.sparse_sizes()}"
    
    # Check that permutation contains all indices
    perm_set = set(perm.tolist())
    expected_set = {0, 1, 2}
    assert perm_set == expected_set, f"Expected permutation indices {expected_set}, got {perm_set}"


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_reverse_cuthill_mckee_symmetric(dtype, device):
    """Test reverse Cuthill-McKee with symmetric matrix."""
    skip_if_unsupported_dtype(dtype, device)
    set_testing_device(device)
    
    # Create a symmetric graph
    row = paddle.to_tensor([0, 1, 1, 0], dtype='int64')
    col = paddle.to_tensor([1, 0, 2, 2], dtype='int64')
    value = tensor([1, 1, 1, 1], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    reordered, perm = reverse_cuthill_mckee(sparse_tensor, is_symmetric=True)
    
    # Check basic properties
    assert len(perm) == 3, f"Expected permutation length 3, got {len(perm)}"
    assert reordered.sparse_sizes() == (3, 3), f"Expected size (3, 3), got {reordered.sparse_sizes()}"


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_sparse_tensor_reverse_cuthill_mckee_method(dtype, device):
    """Test SparseTensor.reverse_cuthill_mckee method."""
    skip_if_unsupported_dtype(dtype, device)
    set_testing_device(device)
    
    # Create a simple graph
    row = paddle.to_tensor([0, 1, 2], dtype='int64')
    col = paddle.to_tensor([1, 2, 0], dtype='int64')
    value = tensor([1, 1, 1], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    # Test method call
    reordered, perm = sparse_tensor.reverse_cuthill_mckee()
    
    # Check basic properties
    assert len(perm) == 3, f"Expected permutation length 3, got {len(perm)}"
    assert reordered.sparse_sizes() == (3, 3), f"Expected size (3, 3), got {reordered.sparse_sizes()}"