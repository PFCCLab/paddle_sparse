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
