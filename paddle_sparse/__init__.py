import os.path as osp

import paddle


__version__ = "0.6.18"

try:
    import paddle_sparse_ops  # noqa
except ImportError:
    raise ImportError(
        f"Could not import `paddle_sparse_ops` in {osp.dirname(__file__)}."
        f"Please run `python setup_ops.py install` manually."
    )


cuda_version = paddle_sparse_ops.cuda_version().item()
if paddle.version.cuda_version is not None and cuda_version != -1:  # pragma: no cover
    if cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in paddle.version.cuda_version.split(".")]

    if t_major != major:
        raise RuntimeError(
            f"Detected that Paddle and paddle_sparse were compiled with "
            f"different CUDA versions. Paddle has CUDA version "
            f"{t_major}.{t_minor} and paddle_sparse has CUDA version "
            f"{major}.{minor}. Please reinstall the paddle_sparse that "
            f"matches your Paddle install."
        )

from .storage import SparseStorage  # noqa
from .tensor import SparseTensor  # noqa
from .narrow import narrow, __narrow_diag__  # noqa
from .select import select  # noqa
from .index_select import index_select, index_select_nnz  # noqa
from .masked_select import masked_select, masked_select_nnz  # noqa
from .permute import permute  # noqa
from .add import add, add_, add_nnz, add_nnz_  # noqa
from .mul import mul, mul_, mul_nnz, mul_nnz_  # noqa
from .reduce import sum, mean, min, max  # noqa
from .cat import cat  # noqa

from .convert import to_paddle_sparse, from_paddle_sparse  # noqa
from .convert import to_scipy, from_scipy  # noqa
from .coalesce import coalesce  # noqa
from .transpose import transpose  # noqa
from .eye import eye  # noqa

__all__ = [
    "SparseStorage",
    "SparseTensor",
    "narrow",
    "__narrow_diag__",
    "select",
    "index_select",
    "index_select_nnz",
    "masked_select",
    "masked_select_nnz",
    "permute",
    "add",
    "add_",
    "add_nnz",
    "add_nnz_",
    "mul",
    "mul_",
    "mul_nnz",
    "mul_nnz_",
    "sum",
    "mean",
    "min",
    "max",
    "cat",
    "to_paddle_sparse",
    "from_paddle_sparse",
    "to_scipy",
    "from_scipy",
    "coalesce",
    "transpose",
    "eye",
    "__version__",
]
