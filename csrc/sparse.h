#pragma once

#include "extensions.h"
#include "macros.h"

SPARSE_API std::vector<paddle::Tensor> cuda_version();

SPARSE_API std::vector<paddle::Tensor> ind2ptr(paddle::Tensor& ind, int64_t M);
SPARSE_API std::vector<paddle::Tensor> ptr2ind(paddle::Tensor& ptr, int64_t E);
