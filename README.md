vectorize
=========

SSE/AVX vectorization wrapper. It supports these functions: (T:float / double)

```
// dst[i] += c * src[i]
template<typename T>
void muladd(const T* src, T c, unsigned int size, T* dst)

// dst = sum(s1[i] * s2[i])
template<typename T>
T dot(const T* s1, const T* s2, unsigned int size)

// dst[i] += src[i]
template<typename T>
void reduce(const T* src, unsigned int size, T* dst)
```
and provides basic SSE/AVX operations wrapper to enable uniform access between sse/avx.

Usage
========

define SIMD types and include product.h.

| define        | SIMD          |
| ------------- |:-------------:|
| #define USE_AVX       | AVX(ver.1) |
| #define USE_SSE       | SSE(ver.2) |
| (nothing)     | none (generic implementation used) |

Example
========

```
#define USE_AVX // specify SIMD types before include header
// #define USE_SSE
#include "product.h"

int main(void) {
     double d1[] = {1, 2, 3, 4};
     double d2[] = {3, 0, -2, 2};
     
     double result = vectorize::dot(d1, d2, 4);
     std::cout << "result=" << result; // "5"
}

```
