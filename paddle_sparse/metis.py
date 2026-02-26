from __future__ import annotations

from typing import Optional, Tuple

import paddle
from paddle import Tensor

from paddle_sparse.permute import permute
from paddle_sparse.tensor import SparseTensor


def weight2metis(weight: Tensor) -> Optional[Tensor]:
    if weight.numel() <= 1:
        return None
    
    sorted_weight = paddle.sort(weight)
    diff = sorted_weight[1:] - sorted_weight[:-1]
    if paddle.sum(diff) == 0:
        return None
    weight_min, weight_max = sorted_weight[0], sorted_weight[-1]
    srange = weight_max - weight_min
    min_diff = paddle.min(diff)
    scale = (min_diff / srange).item()
    
    weight_ratio = ((weight - weight_min) / srange * 1000).astype('int64')
    return weight_ratio


def partition(
    src: SparseTensor,
    num_parts: int,
    recursive: bool = False,
    weighted: bool = False,
    node_weight: Optional[Tensor] = None,
    balance_edge: bool = False,
) -> Tuple[SparseTensor, Tensor, Tensor]:
    assert num_parts >= 1
    if num_parts == 1:
        partptr = paddle.to_tensor([0, src.size(0)], dtype='int64')
        perm = paddle.arange(src.size(0), dtype='int64')
        return src, partptr, perm

    if balance_edge and node_weight is not None:
        raise ValueError("Cannot set 'balance_edge' and 'node_weight' at the "
                         "same time in 'partition'")

    n = src.size(0)
    
    if balance_edge:
        row, col, _ = src.coo()
        node_weight = paddle.zeros([n], dtype='int64')
        for i in range(len(col)):
            node_weight[row[i]] += 1

    if node_weight is not None:
        perm = paddle.argsort(node_weight)
    else:
        perm = paddle.arange(n, dtype='int64')
    
    part_size = n // num_parts
    remainder = n % num_parts
    
    partptr = paddle.zeros([num_parts + 1], dtype='int64')
    for i in range(num_parts):
        partptr[i + 1] = partptr[i] + part_size + (1 if i < remainder else 0)
    
    out = permute(src, perm)
    
    cluster = paddle.zeros([n], dtype='int64')
    for i in range(num_parts):
        start_idx = partptr[i].item()
        end_idx = partptr[i + 1].item()
        cluster[start_idx:end_idx] = i
    
    return out, partptr, perm


SparseTensor.partition = partition