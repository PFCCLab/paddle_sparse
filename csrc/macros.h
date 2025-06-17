#pragma once

#define SPARSE_API

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define SPARSE_INLINE_VARIABLE inline
#else
#define SPARSE_INLINE_VARIABLE __attribute__((weak))
#endif
