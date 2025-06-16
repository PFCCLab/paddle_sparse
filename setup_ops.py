import glob
import os
import os.path as osp
import platform
from itertools import product

import paddle
from paddle.utils.cpp_extension import CppExtension
from paddle.utils.cpp_extension import CUDAExtension
from paddle.utils.cpp_extension import setup
from paddle.utils.cpp_extension.cpp_extension import CUDA_HOME

__version__ = "0.6.18"


WITH_CUDA = False
if paddle.device.cuda.device_count() > 0:
    WITH_CUDA = CUDA_HOME is not None
suffices = ["cpu", "cuda"] if WITH_CUDA else ["cpu"]
if os.getenv("FORCE_CUDA", "0") == "1":
    suffices = ["cuda", "cpu"]
if os.getenv("FORCE_ONLY_CUDA", "0") == "1":
    suffices = ["cuda"]
if os.getenv("FORCE_ONLY_CPU", "0") == "1":
    suffices = ["cpu"]

BUILD_DOCS = os.getenv("BUILD_DOCS", "0") == "1"

WITH_METIS = True if os.getenv("WITH_METIS", "0") == "1" else False
WITH_MTMETIS = True if os.getenv("WITH_MTMETIS", "0") == "1" else False

WITH_SYMBOLS = True if os.getenv("WITH_SYMBOLS", "0") == "1" else False

# TODO(beinggod): Need to verify when WITH_METIS=1 and WITH_MTMETIS=1
assert (
    WITH_METIS is False and WITH_MTMETIS is False
), "Not support METIS and MTMETIS now."
assert platform.system() == "Linux", "Only support build on linux now."


def set_cuda_archs():
    major, _ = paddle.version.cuda_version.split(".")
    if int(major) >= 12:
        paddle_known_gpu_archs = [50, 60, 61, 70, 75, 80, 90]
    elif int(major) >= 11:
        paddle_known_gpu_archs = [50, 60, 61, 70, 75, 80]
    if int(major) >= 10:
        paddle_known_gpu_archs = [50, 52, 60, 61, 70, 75]
    else:
        raise ValueError("Not support cuda version.")

    os.environ["PADDLE_CUDA_ARCH_LIST"] = ",".join(
        [str(arch) for arch in paddle_known_gpu_archs]
    )


def get_extensions():
    extensions_dir = osp.join("csrc")
    main_files = glob.glob(osp.join(extensions_dir, "*.cpp"))
    main_files = [path for path in main_files]

    define_macros = [("WITH_PYTHON", None)]
    undef_macros = []

    libraries = []
    if WITH_METIS:
        define_macros += [("WITH_METIS", None)]
        libraries += ["metis"]
    if WITH_MTMETIS:
        define_macros += [("WITH_MTMETIS", None)]
        define_macros += [("MTMETIS_64BIT_VERTICES", None)]
        define_macros += [("MTMETIS_64BIT_EDGES", None)]
        define_macros += [("MTMETIS_64BIT_WEIGHTS", None)]
        define_macros += [("MTMETIS_64BIT_PARTITIONS", None)]
        libraries += ["mtmetis", "wildriver"]

    extra_link_args = [] if WITH_SYMBOLS else ["-s"]

    extra_compile_args = {"cxx": ["-O3", "-Wno-sign-compare", "-fopenmp"]}
    if "cuda" in suffices:
        set_cuda_archs()
        define_macros += [("WITH_CUDA", None)]
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
        nvcc_flags += ["-O3"]
        nvcc_flags += ["--expt-relaxed-constexpr"]
        extra_compile_args["nvcc"] = nvcc_flags

    for main, suffix in product(main_files, suffices):
        name = main.split(os.sep)[-1][:-4]
        sources = [main]

        path = osp.join(extensions_dir, "cpu", f"{name}_cpu.cpp")
        if osp.exists(path):
            sources += [path]

        path = osp.join(extensions_dir, "cuda", f"{name}_cuda.cu")
        if suffix == "cuda" and osp.exists(path):
            sources += [path]

    phmap_dir = osp.abspath("third_party/parallel-hashmap")

    Extension = CUDAExtension if "cuda" in suffices else CppExtension
    extension = Extension(
        sources,
        include_dirs=[extensions_dir, phmap_dir],
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
    )

    return extension


setup(
    name="paddle_sparse_ops",
    version=__version__,
    description=(
        "Paddle Custom Operators Extension Library of Optimized Autograd Sparse "
        "Matrix Operations. "
    ),
    author="Ruibin Cheung",
    author_email="beinggod@foxmail.com",
    python_requires=">=3.8",
    ext_modules=get_extensions(),
)
