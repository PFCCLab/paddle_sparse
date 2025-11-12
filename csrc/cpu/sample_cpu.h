#pragma once

#include "../extensions.h"

std::tuple<paddle::Tensor, paddle::Tensor, paddle::Tensor, paddle::Tensor>
sample_adj_cpu(paddle::Tensor rowptr,
               paddle::Tensor col,
               paddle::Tensor idx,
               int64_t num_neighbors,
               bool replace);
