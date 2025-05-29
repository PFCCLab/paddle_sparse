#pragma once

#include "../extensions.h"

#define CHECK_CUDA(x) PD_CHECK(x.is_gpu(), #x " must be CUDA tensor")
#define CHECK_INPUT(x) PD_CHECK(x, "Input mismatch")
