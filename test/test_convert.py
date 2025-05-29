import paddle

from paddle_sparse import from_paddle_sparse
from paddle_sparse import from_scipy
from paddle_sparse import to_paddle_sparse
from paddle_sparse import to_scipy


def test_convert_scipy():
    index = paddle.to_tensor(
        [[0, 0, 1, 2, 2], [0, 2, 1, 0, 1]], place=paddle.CPUPlace()
    )
    value = paddle.to_tensor([1, 2, 4, 1, 3], place=paddle.CPUPlace())
    N = 3

    out = from_scipy(to_scipy(index, value, N, N))
    assert out[0].tolist() == index.tolist()
    assert out[1].tolist() == value.tolist()


def test_convert_paddle_sparse():
    index = paddle.to_tensor([[0, 0, 1, 2, 2], [0, 2, 1, 0, 1]])
    value = paddle.to_tensor([1, 2, 4, 1, 3])
    N = 3

    out = from_paddle_sparse(to_paddle_sparse(index, value, N, N).coalesce())
    assert out[0].tolist() == index.tolist()
    assert out[1].tolist() == value.tolist()
