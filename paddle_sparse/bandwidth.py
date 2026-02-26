from __future__ import annotations

from typing import Tuple, Optional

import paddle
from paddle_sparse.tensor import SparseTensor
from paddle_sparse.permute import permute


def reverse_cuthill_mckee(src: SparseTensor, 
                         is_symmetric: Optional[bool] = None) -> Tuple[SparseTensor, paddle.Tensor]:
    if is_symmetric is None:
        is_symmetric = src.is_symmetric()

    if not is_symmetric:
        src = src.to_symmetric()

    try:
        import scipy.sparse as sp
        
        sp_src = src.to_scipy(layout='csr')
        
        perm = sp.csgraph.reverse_cuthill_mckee(sp_src, symmetric_mode=True).copy()
        perm = paddle.to_tensor(perm, dtype='int64')
        
        out = permute(src, perm)
        
        return out, perm
        
    except ImportError:
        print("Warning: scipy not available, using simple bandwidth reduction")
        
        row, col, _ = src.coo()
        n = src.size(0)
        
        degrees = paddle.zeros([n], dtype='int64')
        for i in range(len(row)):
            degrees[row[i]] += 1
            if row[i] != col[i]:
                degrees[col[i]] += 1
        
        perm = paddle.argsort(degrees)
        
        out = permute(src, perm)
        
        return out, perm


SparseTensor.reverse_cuthill_mckee = reverse_cuthill_mckee