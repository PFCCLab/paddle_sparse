#pragma once

#include "../extensions.h"
#include "parallel_hashmap/phmap.h"

#define CHECK_CPU(x) PD_CHECK(x.is_cpu(), #x " must be CPU tensor")
#define CHECK_INPUT(x) PD_CHECK(x, "Input mismatch")
#define CHECK_LT(low, high) \
  PD_CHECK(low < high, "low must be smaller than high")

#define PD_DISPATCH_HAS_VALUE(optional_value, ...) \
  [&] {                                            \
    if (optional_value.has_value()) {              \
      const bool HAS_VALUE = true;                 \
      return __VA_ARGS__();                        \
    } else {                                       \
      const bool HAS_VALUE = false;                \
      return __VA_ARGS__();                        \
    }                                              \
  }()

inline int64_t uniform_randint(int64_t low,
                               int64_t high,
                               const paddle::Place &place) {
  CHECK_LT(low, high);
  auto ret = paddle::experimental::randint(
      low, high, {1}, paddle::DataType::INT64, place);
  auto ptr = ret.data<int64_t>();
  return *ptr;
}

inline int64_t uniform_randint(int64_t high, const paddle::Place &place) {
  return uniform_randint(0, high, place);
}
