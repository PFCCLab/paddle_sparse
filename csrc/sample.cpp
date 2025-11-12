#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <paddle/extension.h>

#include "cpu/sample_cpu.h"

SPARSE_API std::vector<paddle::Tensor> sample_adj(paddle::Tensor &rowptr,
                                                  paddle::Tensor &col,
                                                  paddle::Tensor &idx,
                                                  int64_t num_neighbors,
                                                  bool replace) {
  if (rowptr.is_gpu()) {
#ifdef WITH_CUDA
    PD_THROW("No CUDA version supported");
#else
    PD_THROW("Not compiled with CUDA support");
#endif
  } else {
    auto ret = sample_adj_cpu(rowptr, col, idx, num_neighbors, replace);
    return {
        std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret)};
  }
}


std::vector<paddle::DataType> sample_adj_infer_dtype(
    const paddle::DataType rowptr_dtype,
    const paddle::DataType col_dtype,
    const paddle::DataType idx_dtype) {
  return {rowptr_dtype, col_dtype, col_dtype, col_dtype};
}

// NOTE: Ignore infer shape because output's shape is dynamic.
PD_BUILD_OP(sample_adj)
    .Inputs({"rowptr", "col", "idx"})
    .Outputs({"out_rowptr", "out_col", "out_n_id", "out_e_id"})
    .Attrs({"num_neighbors: int64_t", "replace: bool"})
    .SetKernelFn(PD_KERNEL(sample_adj))
    .SetInferDtypeFn(PD_INFER_DTYPE(sample_adj_infer_dtype));
