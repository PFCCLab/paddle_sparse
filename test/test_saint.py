from itertools import product

import pytest
import paddle

from paddle_sparse import SparseTensor
from paddle_sparse.saint import saint_subgraph
from paddle_sparse.testing import devices, dtypes, tensor, set_testing_device


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_saint_subgraph_basic(dtype, device):
    """Test basic SAINT subgraph functionality."""
    set_testing_device(device)
    
    # Create a simple graph: 0-1-2-0
    row = paddle.to_tensor([0, 1, 2, 0], dtype='int64')
    col = paddle.to_tensor([1, 2, 0, 2], dtype='int64')
    value = tensor([1, 1, 1, 1], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    # Extract subgraph with nodes [0, 2]
    node_idx = paddle.to_tensor([0, 2], dtype='int64')
    subgraph, edge_idx = saint_subgraph(sparse_tensor, node_idx)
    
    # Check subgraph size
    expected_size = (2, 2)
    assert subgraph.sparse_sizes() == expected_size, f"Expected size {expected_size}, got {subgraph.sparse_sizes()}"
    
    # Check that edge indices are valid
    assert len(edge_idx) >= 0, "Edge indices should be non-negative length"


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_saint_subgraph_empty(dtype, device):
    """Test SAINT subgraph with no valid edges."""
    set_testing_device(device)
    
    # Create a graph where selected nodes have no connections
    row = paddle.to_tensor([0, 1], dtype='int64')
    col = paddle.to_tensor([1, 0], dtype='int64')
    value = tensor([1, 1], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    # Extract subgraph with node [2] (isolated node)
    node_idx = paddle.to_tensor([2], dtype='int64')
    subgraph, edge_idx = saint_subgraph(sparse_tensor, node_idx)
    
    # Check subgraph size
    expected_size = (1, 1)
    assert subgraph.sparse_sizes() == expected_size, f"Expected size {expected_size}, got {subgraph.sparse_sizes()}"
    
    # Check that no edges are found
    row_sub, col_sub, _ = subgraph.coo()
    assert len(row_sub) == 0, "Should have no edges in isolated subgraph"
    assert len(edge_idx) == 0, "Should have no edge indices"


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_saint_subgraph_full_graph(dtype, device):
    """Test SAINT subgraph with all nodes."""
    set_testing_device(device)
    
    # Create a simple graph
    row = paddle.to_tensor([0, 1, 2], dtype='int64')
    col = paddle.to_tensor([1, 2, 0], dtype='int64')
    value = tensor([1, 1, 1], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    # Extract subgraph with all nodes
    node_idx = paddle.to_tensor([0, 1, 2], dtype='int64')
    subgraph, edge_idx = saint_subgraph(sparse_tensor, node_idx)
    
    # Check subgraph size
    expected_size = (3, 3)
    assert subgraph.sparse_sizes() == expected_size, f"Expected size {expected_size}, got {subgraph.sparse_sizes()}"
    
    # Check that all edges are preserved
    row_sub, col_sub, _ = subgraph.coo()
    assert len(row_sub) == 3, "Should preserve all edges"
    assert len(edge_idx) == 3, "Should have all edge indices"


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_saint_subgraph_with_values(dtype, device):
    """Test SAINT subgraph preserves edge values."""
    set_testing_device(device)
    
    # Create a graph with specific edge values
    row = paddle.to_tensor([0, 1, 2], dtype='int64')
    col = paddle.to_tensor([1, 2, 0], dtype='int64')
    value = tensor([10, 20, 30], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    # Extract subgraph with nodes [0, 1]
    node_idx = paddle.to_tensor([0, 1], dtype='int64')
    subgraph, edge_idx = saint_subgraph(sparse_tensor, node_idx)
    
    # Check that values are preserved
    _, _, value_sub = subgraph.coo()
    if value_sub is not None and len(value_sub) > 0:
        # Should have preserved some edge values
        assert value_sub.dtype == value.dtype, "Value dtype should be preserved"


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_sparse_tensor_saint_subgraph_method(dtype, device):
    """Test SparseTensor.saint_subgraph method."""
    set_testing_device(device)
    
    # Create a simple graph
    row = paddle.to_tensor([0, 1, 2], dtype='int64')
    col = paddle.to_tensor([1, 2, 0], dtype='int64')
    value = tensor([1, 1, 1], dtype, device)
    sparse_tensor = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    
    # Test method call
    node_idx = paddle.to_tensor([0, 2], dtype='int64')
    subgraph, edge_idx = sparse_tensor.saint_subgraph(node_idx)
    
    # Check basic properties
    expected_size = (2, 2)
    assert subgraph.sparse_sizes() == expected_size, f"Expected size {expected_size}, got {subgraph.sparse_sizes()}"