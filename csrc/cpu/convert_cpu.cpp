#include "convert_cpu.h"

#include "utils.h"


paddle::Tensor ind2ptr_cpu(paddle::Tensor ind, int64_t M) {
  CHECK_CPU(ind);

  if (ind.numel() == 0) {
    return paddle::zeros({M + 1}, ind.dtype(), ind.place());
  }

  auto out = paddle::empty({M + 1}, ind.dtype(), ind.place());
  auto ind_data = ind.data<int64_t>();
  auto out_data = out.data<int64_t>();

  int64_t numel = ind.numel();

  for (int64_t i = 0; i <= ind_data[0]; i++) out_data[i] = 0;

  int64_t idx = ind_data[0], next_idx;
  for (int64_t i = 0; i < numel - 1; i++) {
    next_idx = ind_data[i + 1];
    for (; idx < next_idx; idx++) out_data[idx + 1] = i + 1;
  }

  for (int64_t i = ind_data[numel - 1] + 1; i < M + 1; i++) out_data[i] = numel;

  return out;
}

paddle::Tensor ptr2ind_cpu(paddle::Tensor ptr, int64_t E) {
  CHECK_CPU(ptr);
  auto out = paddle::empty({E}, ptr.dtype(), ptr.place());
  auto ptr_data = ptr.data<int64_t>();
  auto out_data = out.data<int64_t>();

  int64_t numel = ptr.numel();

  int64_t idx = ptr_data[0], next_idx;
  for (int64_t i = 0; i < numel - 1; i++) {
    next_idx = ptr_data[i + 1];
    for (int64_t e = idx; e < next_idx; e++) out_data[e] = i;
    idx = next_idx;
  }

  return out;
}
