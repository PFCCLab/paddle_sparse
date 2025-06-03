#pragma once

#include "../extensions.h"

paddle::Tensor ind2ptr_cuda(paddle::Tensor ind, int64_t M);
paddle::Tensor ptr2ind_cuda(paddle::Tensor ptr, int64_t E);
