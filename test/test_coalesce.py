import paddle

from paddle_sparse import coalesce


def test_coalesce():
    row = paddle.to_tensor([1, 0, 1, 0, 2, 1])
    col = paddle.to_tensor([0, 1, 1, 1, 0, 0])
    index = paddle.stack([row, col], axis=0)

    index, _ = coalesce(index, None, m=3, n=2)
    assert index.tolist() == [[0, 1, 1, 2], [1, 0, 1, 0]]


def test_coalesce_add():
    row = paddle.to_tensor([1, 0, 1, 0, 2, 1])
    col = paddle.to_tensor([0, 1, 1, 1, 0, 0])
    index = paddle.stack([row, col], axis=0)
    value = paddle.to_tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])

    index, value = coalesce(index, value, m=3, n=2)
    assert index.tolist() == [[0, 1, 1, 2], [1, 0, 1, 0]]
    assert value.tolist() == [[6, 8], [7, 9], [3, 4], [5, 6]]


def test_coalesce_max():
    row = paddle.to_tensor([1, 0, 1, 0, 2, 1])
    col = paddle.to_tensor([0, 1, 1, 1, 0, 0])
    index = paddle.stack([row, col], axis=0)
    value = paddle.to_tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])

    index, value = coalesce(index, value, m=3, n=2, op="max")
    assert index.tolist() == [[0, 1, 1, 2], [1, 0, 1, 0]]
    assert value.tolist() == [[4, 5], [6, 7], [3, 4], [5, 6]]
