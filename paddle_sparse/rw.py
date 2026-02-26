from __future__ import annotations

import paddle
from paddle import Tensor

from paddle_sparse.tensor import SparseTensor


def random_walk(src: SparseTensor, start: Tensor, walk_length: int) -> Tensor:
    rowptr, col, _ = src.csr()
    
    batch_size = start.shape[0]
    walks = paddle.zeros([batch_size, walk_length + 1], dtype='int64')
    walks[:, 0] = start
    
    current_nodes = start.clone()
    
    for step in range(walk_length):
        next_nodes = paddle.zeros_like(current_nodes)
        
        for i in range(batch_size):
            node = current_nodes[i].item()
            
            if node < len(rowptr) - 1:
                start_idx = rowptr[node].item()
                end_idx = rowptr[node + 1].item()
                
                if end_idx > start_idx:
                    neighbors = col[start_idx:end_idx]
                    
                    if len(neighbors) > 0:
                        rand_idx = paddle.randint(0, len(neighbors), [1])
                        next_nodes[i] = neighbors[rand_idx[0]]
                    else:
                        next_nodes[i] = current_nodes[i]
                else:
                    next_nodes[i] = current_nodes[i]
            else:
                next_nodes[i] = current_nodes[i]
        
        walks[:, step + 1] = next_nodes
        current_nodes = next_nodes
    
    return walks


SparseTensor.random_walk = random_walk