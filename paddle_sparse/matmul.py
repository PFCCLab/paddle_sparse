from __future__ import annotations

from typing import Optional, Tuple

import paddle
from paddle import Tensor

from paddle_sparse.tensor import SparseTensor


def spmm_sum(src: SparseTensor, other: paddle.Tensor) -> paddle.Tensor:
    rowptr, col, value = src.csr()
    
    if value is not None:
        value = value.astype(other.dtype)
    
    # Use scatter operations for aggregation
    row = src.storage.row()
    out = paddle.index_select(other, col, axis=-2)
    
    if value is not None:
        out = out * value.unsqueeze(-1)
    
    try:
        from paddle_scatter import scatter_add
        out = scatter_add(out, row, dim=-2, dim_size=src.size(0))
    except ImportError:
        def scatter_add_fallback(src, index, dim, dim_size):
            if dim == -2:
                dim = src.ndim - 2
            out_shape = list(src.shape)
            out_shape[dim] = dim_size
            out = paddle.zeros(out_shape, dtype=src.dtype)
            for i in range(index.shape[0]):
                idx = index[i].item()
                out[idx] += src[i]
            return out
        out = scatter_add_fallback(out, row, dim=-2, dim_size=src.size(0))
    
    return out


def spmm_add(src: SparseTensor, other: paddle.Tensor) -> paddle.Tensor:
    return spmm_sum(src, other)


def spmm_mean(src: SparseTensor, other: paddle.Tensor) -> paddle.Tensor:
    rowptr, col, value = src.csr()
    
    if value is not None:
        value = value.astype(other.dtype)
    
    row = src.storage.row()
    out = paddle.index_select(other, col, axis=-2)
    
    if value is not None:
        out = out * value.unsqueeze(-1)
    
    try:
        from paddle_scatter import scatter_mean
        out = scatter_mean(out, row, dim=-2, dim_size=src.size(0))
    except ImportError:
        def scatter_mean_fallback(src, index, dim, dim_size):
            if dim == -2:
                dim = src.ndim - 2
            out_shape = list(src.shape)
            out_shape[dim] = dim_size
            out = paddle.zeros(out_shape, dtype=src.dtype)
            count = paddle.zeros([dim_size], dtype=src.dtype)
            
            for i in range(index.shape[0]):
                idx = index[i].item()
                count[idx] += 1.0
                out[idx] += src[i]
            
            count = paddle.maximum(count, paddle.ones_like(count))
            out = out / count.unsqueeze(-1)
            return out
        out = scatter_mean_fallback(out, row, dim=-2, dim_size=src.size(0))
    
    return out


def spmm_min(src: SparseTensor, other: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
    rowptr, col, value = src.csr()
    
    if value is not None:
        value = value.astype(other.dtype)
    
    row = src.storage.row()
    out = paddle.index_select(other, col, axis=-2)
    
    if value is not None:
        out = out * value.unsqueeze(-1)
    
    try:
        from paddle_scatter import scatter_min
        out, argmin = scatter_min(out, row, dim=-2, dim_size=src.size(0))
    except ImportError:
        def scatter_min_fallback(src, index, dim, dim_size):
            if dim == -2:
                dim = src.ndim - 2
            out_shape = list(src.shape)
            out_shape[dim] = dim_size
            out = paddle.full(out_shape, float('inf'), dtype=src.dtype)
            argmin = paddle.zeros([dim_size], dtype='int64')
            
            for i in range(index.shape[0]):
                idx = index[i].item()
                mask = src[i] < out[idx]
                if paddle.any(mask):
                    out[idx] = paddle.where(mask, src[i], out[idx])
                    argmin[idx] = i
            
            return out, argmin
        out, argmin = scatter_min_fallback(out, row, dim=-2, dim_size=src.size(0))
    
    return out, argmin


def spmm_max(src: SparseTensor, other: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
    rowptr, col, value = src.csr()
    
    if value is not None:
        value = value.astype(other.dtype)
    
    row = src.storage.row()
    out = paddle.index_select(other, col, axis=-2)
    
    if value is not None:
        out = out * value.unsqueeze(-1)
    
    try:
        from paddle_scatter import scatter_max
        out, argmax = scatter_max(out, row, dim=-2, dim_size=src.size(0))
    except ImportError:
        def scatter_max_fallback(src, index, dim, dim_size):
            if dim == -2:
                dim = src.ndim - 2
            out_shape = list(src.shape)
            out_shape[dim] = dim_size
            out = paddle.full(out_shape, float('-inf'), dtype=src.dtype)
            argmax = paddle.zeros([dim_size], dtype='int64')
            
            for i in range(index.shape[0]):
                idx = index[i].item()
                mask = src[i] > out[idx]
                if paddle.any(mask):
                    out[idx] = paddle.where(mask, src[i], out[idx])
                    argmax[idx] = i
            
            return out, argmax
        out, argmax = scatter_max_fallback(out, row, dim=-2, dim_size=src.size(0))
    
    return out, argmax


def spmm(src: SparseTensor, other: paddle.Tensor, reduce: str = "sum") -> paddle.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return spmm_sum(src, other)
    elif reduce == 'mean':
        return spmm_mean(src, other)
    elif reduce == 'min':
        return spmm_min(src, other)[0]
    elif reduce == 'max':
        return spmm_max(src, other)[0]
    else:
        raise ValueError


def spspmm_sum(src: SparseTensor, other: SparseTensor) -> SparseTensor:
    # Use paddle.sparse.matmul on GPU, fallback to manual implementation on CPU
    device = src.device()
    is_gpu = 'gpu' in str(device).lower()

    if is_gpu:
        # Use Paddle's built-in sparse matrix multiplication on GPU
        try:
            rowA, colA, valueA = src.coo()
            rowB, colB, valueB = other.coo()
            
            A_coo = paddle.sparse.sparse_coo_tensor(
                paddle.stack([rowA, colA]),
                valueA if valueA is not None else paddle.ones([len(rowA)], dtype='float32'),
                src.sparse_sizes(),
                place=device
            )
            B_coo = paddle.sparse.sparse_coo_tensor(
                paddle.stack([rowB, colB]),
                valueB if valueB is not None else paddle.ones([len(rowB)], dtype='float32'),
                other.sparse_sizes(),
                place=device
            )
            
            C_coo = paddle.sparse.matmul(A_coo, B_coo)
            
            indices = C_coo.indices()
            row, col = indices[0], indices[1]
            value = C_coo.values() if C_coo.values() is not None else None
            
            return SparseTensor(
                row=row,
                col=col,
                value=value,
                sparse_sizes=(C_coo.shape[0], C_coo.shape[1]),
                is_sorted=True,
                trust_data=True,
            )
        except Exception as e:
            # Fallback to manual implementation if GPU kernel fails
            print(f"Warning: GPU spspmm failed ({e}), falling back to CPU implementation")
    
    # Manual sparse-sparse matrix multiplication implementation (CPU fallback)
    rowA, colA, valueA = src.coo()
    rowB, colB, valueB = other.coo()
    
    # Ensure inputs are coalesced
    if not src.is_coalesced():
        src = src.coalesce()
    if not other.is_coalesced():
        other = other.coalesce()
        rowA, colA, valueA = src.coo()
        rowB, colB, valueB = other.coo()
    
    # Create result storage
    result_dict = {}
    
    # Manual sparse-sparse multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
    for idx_a in range(rowA.shape[0]):
        i, k = rowA[idx_a].item(), colA[idx_a].item()
        for idx_b in range(rowB.shape[0]):
            if rowB[idx_b].item() == k:
                j = colB[idx_b].item()
                key = (i, j)
                if valueA is not None and valueB is not None:
                    val = valueA[idx_a].item() * valueB[idx_b].item()
                else:
                    val = 1.0
                
                if key in result_dict:
                    result_dict[key] += val
                else:
                    result_dict[key] = val
    
    # Convert to tensors
    if result_dict:
        rows = [k[0] for k in result_dict.keys()]
        cols = [k[1] for k in result_dict.keys()]
        values = list(result_dict.values())
        
        # Sort by (row, col) for consistent output
        sorted_indices = sorted(zip(rows, cols, values), key=lambda x: (x[0], x[1]))
        rows = [idx[0] for idx in sorted_indices]
        cols = [idx[1] for idx in sorted_indices]
        values = [idx[2] for idx in sorted_indices]
        
        row = paddle.to_tensor(rows, dtype='int64')
        col = paddle.to_tensor(cols, dtype='int64')
        value = paddle.to_tensor(values, dtype=src.dtype() if src.has_value() else 'float32')
    else:
        row = paddle.to_tensor([], dtype='int64')
        col = paddle.to_tensor([], dtype='int64')
        value = None
    
    m = src.sparse_size(0)
    n = other.sparse_size(1)
    
    return SparseTensor(
        row=row,
        col=col,
        value=value,
        sparse_sizes=(m, n),
        is_sorted=True,
        trust_data=True,
    )


def spspmm_add(src: SparseTensor, other: SparseTensor) -> SparseTensor:
    return spspmm_sum(src, other)


def spspmm(src: SparseTensor, other: SparseTensor, reduce: str = "sum") -> SparseTensor:
    if reduce == 'sum' or reduce == 'add':
        return spspmm_sum(src, other)
    elif reduce == 'mean' or reduce == 'min' or reduce == 'max':
        raise NotImplementedError
    else:
        raise ValueError


def matmul(src, other, reduce="sum"):
    if isinstance(other, paddle.Tensor):
        return spmm(src, other, reduce)
    elif isinstance(other, SparseTensor):
        return spspmm(src, other, reduce)
    raise ValueError


# Add methods to SparseTensor class
SparseTensor.spmm = lambda self, other, reduce="sum": spmm(self, other, reduce)
SparseTensor.spspmm = lambda self, other, reduce="sum": spspmm(self, other, reduce)
SparseTensor.matmul = lambda self, other, reduce="sum": matmul(self, other, reduce)
SparseTensor.__matmul__ = lambda self, other: matmul(self, other, 'sum')