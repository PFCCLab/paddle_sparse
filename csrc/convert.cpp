#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <paddle/extension.h>

#include "cpu/convert_cpu.h"

#ifdef WITH_CUDA
#include "cuda/convert_cuda.h"
#endif


SPARSE_API std::vector<paddle::Tensor> ind2ptr(paddle::Tensor& ind, int64_t M) {
  if (ind.is_gpu()) {
#ifdef WITH_CUDA
    return {ind2ptr_cuda(ind, M)};
#else
    PD_THROW("Not compiled with CUDA support");
#endif
  } else {
    return {ind2ptr_cpu(ind, M)};
  }
}

SPARSE_API std::vector<paddle::Tensor> ptr2ind(paddle::Tensor& ptr, int64_t E) {
  if (ptr.is_gpu()) {
#ifdef WITH_CUDA
    return {ptr2ind_cuda(ptr, E)};
#else
    PD_THROW("Not compiled with CUDA support");
#endif
  } else {
    return {ptr2ind_cpu(ptr, E)};
  }
}


PD_BUILD_OP(ind2ptr)
    .Inputs({"ind"})
    .Outputs({"out"})
    .Attrs({"M: int64_t"})
    .SetKernelFn(PD_KERNEL(ind2ptr));


PD_BUILD_OP(ptr2ind)
    .Inputs({"ptr"})
    .Outputs({"out"})
    .Attrs({"E: int64_t"})
    .SetKernelFn(PD_KERNEL(ptr2ind));
