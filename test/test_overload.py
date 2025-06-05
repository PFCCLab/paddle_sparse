import paddle

from paddle_sparse.tensor import SparseTensor


def test_overload():
    row = paddle.to_tensor([0, 1, 1, 2, 2])
    col = paddle.to_tensor([1, 0, 2, 1, 2])
    mat = SparseTensor(row=row, col=col)

    other = paddle.to_tensor([1, 2, 3]).view([3, 1])
    # TODO: temporary skip this case. More details: https://github.com/PaddlePaddle/Paddle/pull/73119
    # other + mat
    mat + other
    # other * mat
    mat * other

    other = paddle.to_tensor([1, 2, 3]).view([1, 3])
    # TODO: temporary skip this case. More details: https://github.com/PaddlePaddle/Paddle/pull/73119
    # other + mat
    mat + other
    # other * mat
    mat * other
