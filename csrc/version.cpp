#ifdef WITH_PYTHON
#include <Python.h>
#endif

#include <paddle/extension.h>

#ifdef WITH_CUDA
#include <cuda.h>
#endif

#include "macros.h"


SPARSE_API std::vector<paddle::Tensor> cuda_version() {
  auto cpu_place = paddle::CPUPlace();
#ifdef WITH_CUDA
  int64_t version = CUDA_VERSION;
#else
  int64_t version = -1;
#endif
  return {paddle::full({1}, version, paddle::DataType::INT64, cpu_place)};
}


std::vector<paddle::DataType> cuda_version_infer_dtype() {
  return {paddle::DataType::INT64};
}


std::vector<std::vector<int64_t>> cuda_version_infer_shape(int64_t M) {
  return {{1}};
}


PD_BUILD_OP(cuda_version)
    .Inputs({})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(cuda_version))
    .SetInferShapeFn(PD_INFER_SHAPE(cuda_version_infer_shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(cuda_version_infer_dtype));
