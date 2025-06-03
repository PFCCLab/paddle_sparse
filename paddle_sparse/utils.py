from typing import Any
from typing import Optional
from typing import Tuple

import paddle
from typing_extensions import Final  # noqa

import paddle_sparse.typing
from paddle_sparse.typing import pyg_lib


def index_sort(
    inputs: paddle.Tensor,
    max_value: Optional[int] = None,
    with_sorted_inputs: Optional[bool] = False,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    r"""See pyg-lib documentation for more details:
    https://pyg-lib.readthedocs.io/en/latest/modules/ops.html"""
    if not paddle_sparse.typing.WITH_INDEX_SORT:  # pragma: no cover
        return (inputs.sort() if with_sorted_inputs else None, inputs.argsort())
    return pyg_lib.ops.index_sort(inputs, max_value)


def is_scalar(other: Any) -> bool:
    return isinstance(other, int) or isinstance(other, float)


def is_pinned_tensor(x: paddle.Tensor) -> bool:
    return "pinned" in str(x.place)
