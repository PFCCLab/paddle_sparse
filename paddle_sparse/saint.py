from __future__ import annotations

from typing import Tuple

import paddle
from paddle_sparse.tensor import SparseTensor


def saint_subgraph(src: SparseTensor, node_idx: paddle.Tensor) -> Tuple[SparseTensor, paddle.Tensor]:
    row, col, value = src.coo()
    
    node_map = paddle.full([src.size(0)], -1, dtype='int64')
    for i, node in enumerate(node_idx):
        node_map[node] = i
    
    valid_edges = []
    edge_indices = []
    
    for i in range(len(row)):
        src_node = row[i].item()
        dst_node = col[i].item()
        
        if node_map[src_node] != -1 and node_map[dst_node] != -1:
            valid_edges.append(i)
            edge_indices.append(i)
    
    if len(valid_edges) == 0:
        new_row = paddle.zeros([0], dtype='int64')
        new_col = paddle.zeros([0], dtype='int64')
        new_value = None if value is None else paddle.zeros([0], dtype=value.dtype)
        edge_index = paddle.zeros([0], dtype='int64')
    else:
        valid_edges = paddle.to_tensor(valid_edges, dtype='int64')
        new_row = paddle.gather(row, valid_edges)
        new_col = paddle.gather(col, valid_edges)
        
        new_row = paddle.gather(node_map, new_row)
        new_col = paddle.gather(node_map, new_col)
        
        if value is not None:
            # paddle.gather doesn't support float16, so convert to float32 temporarily
            original_dtype = value.dtype
            if original_dtype == paddle.float16:
                value_float32 = value.cast('float32')
                new_value = paddle.gather(value_float32, valid_edges).cast(original_dtype)
            else:
                new_value = paddle.gather(value, valid_edges)
        else:
            new_value = None
            
        edge_index = valid_edges

    subgraph_size = len(node_idx)
    out = SparseTensor(
        row=new_row, 
        rowptr=None, 
        col=new_col, 
        value=new_value,
        sparse_sizes=(subgraph_size, subgraph_size),
        is_sorted=True
    )

    return out, edge_index


SparseTensor.saint_subgraph = saint_subgraph