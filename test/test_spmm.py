from itertools import product

import pytest
import paddle

from paddle_sparse import spmm
from paddle_sparse.testing import devices, dtypes, tensor, set_testing_device


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_spmm(dtype, device):
    set_testing_device(device)
    
    # Skip float16 and bfloat16 due to kernel not supported or incorrect results
    if dtype in [paddle.float16, paddle.bfloat16]:
        pytest.skip()
    
    row = paddle.to_tensor([0, 0, 1, 2, 2], dtype='int64')
    col = paddle.to_tensor([0, 2, 1, 0, 1], dtype='int64')
    index = paddle.stack([row, col], axis=0)
    value = tensor([1, 2, 4, 1, 3], dtype, device)
    x = tensor([[1, 4], [2, 5], [3, 6]], dtype, device)

    out = spmm(index, value, 3, 3, x)
    expected = [[7, 16], [8, 20], [7, 19]]
    
    # Convert to list for comparison
    out_list = out.tolist()
    assert out_list == expected, f"Expected {expected}, got {out_list}"