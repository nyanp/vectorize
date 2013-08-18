#pragma once
#include <immintrin.h>
#include <cstdint>
#include <cassert>
#include <numeric>

namespace vectorize {
namespace detail {

// traits

template <typename T>
struct generic {
    typedef T register_type;
    typedef T value_type;
    enum {
        unroll_size = 1
    };
    static register_type set1(const value_type& x) { return x; }
    static register_type zero() { return 0.0; }
    static register_type mul(const register_type& v1, const register_type& v2) { return v1 * v2; }
    static register_type add(const register_type& v1, const register_type& v2) { return v1 + v2; }
    static register_type load(const value_type* px) { return *px; }
    static register_type loadu(const value_type* px) { return *px; }
    static void store(value_type* px, const register_type& v) { *px = v; }
    static void storeu(value_type* px, const register_type& v) { *px = v; }
    static value_type resemble(const register_type& x) { return x; }
};

#ifdef USE_SSE

struct float_sse {
    typedef __m128 register_type;
    typedef float value_type;
    enum {
        unroll_size = 4
    };
    static register_type set1(const value_type& x) { return _mm_set1_ps(x); }
    static register_type zero() { register_type v = {}; return v; }
    static register_type mul(const register_type& v1, const register_type& v2) { return _mm_mul_ps(v1, v2); }
    static register_type add(const register_type& v1, const register_type& v2) { return _mm_add_ps(v1, v2); }
    static register_type load(const value_type* px) { return _mm_load_ps(px); }
    static register_type loadu(const value_type* px) { return _mm_loadu_ps(px); }
    static void store(value_type* px, const register_type& v) { _mm_store_ps(px, v); }
    static void storeu(value_type* px, const register_type& v) { _mm_storeu_ps(px, v); }
    static value_type resemble(const register_type& x) {
        __declspec(align(16)) float tmp[4];
        _mm_store_ps(tmp, x);
        return tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
};

struct double_sse {
    typedef __m128d register_type;
    typedef double value_type;
    enum {
        unroll_size = 2
    };
    static register_type set1(const value_type& x) { return _mm_set1_pd(x); }
    static register_type zero() { register_type v = {}; return v; }
    static register_type mul(const register_type& v1, const register_type& v2) { return _mm_mul_pd(v1, v2); }
    static register_type add(const register_type& v1, const register_type& v2) { return _mm_add_pd(v1, v2); }
    static register_type load(const value_type* px) { return _mm_load_pd(px); }
    static register_type loadu(const value_type* px) { return _mm_loadu_pd(px); }
    static void store(value_type* px, const register_type& v) { _mm_store_pd(px, v); }
    static void storeu(value_type* px, const register_type& v) { _mm_storeu_pd(px, v); }
    static value_type resemble(const register_type& x) {
        __declspec(align(16)) double tmp[2];
        _mm_store_pd(tmp, x);
        return tmp[0] + tmp[1];
    }
};

template<typename T>
struct sse {};
template<>
struct sse<float> : public float_sse {};
template<>
struct sse<double> : public double_sse {};

#endif // USE_SSE

#ifdef USE_AVX

struct float_avx {
    typedef __m256 register_type;
    typedef float value_type;
    enum {
        unroll_size = 8
    };
    static register_type set1(const value_type& x) { return _mm256_set1_ps(x); }
    static register_type zero() { register_type v = {}; return v; }
    static register_type mul(const register_type& v1, const register_type& v2) { return _mm256_mul_ps(v1, v2); }
    static register_type add(const register_type& v1, const register_type& v2) { return _mm256_add_ps(v1, v2); }
    static register_type load(const value_type* px) { return _mm256_load_ps(px); }
    static register_type loadu(const value_type* px) { return _mm256_loadu_ps(px); }
    static void store(value_type* px, const register_type& v) { _mm256_store_ps(px, v); }
    static void storeu(value_type* px, const register_type& v) { _mm256_storeu_ps(px, v); }
    static value_type resemble(const register_type& x) { 
        __declspec(align(32)) float tmp[8];
        _mm256_store_ps(tmp, x);
        return std::accumulate(tmp, tmp + 8, 0.0f);
    }
};

struct double_avx {
    typedef __m256d register_type;
    typedef double value_type;
    enum {
        unroll_size = 4
    };
    static register_type set1(const value_type& x) { return _mm256_set1_pd(x); }
    static register_type zero() { register_type v = {}; return v; }
    static register_type mul(const register_type& v1, const register_type& v2) { return _mm256_mul_pd(v1, v2); }
    static register_type add(const register_type& v1, const register_type& v2) { return _mm256_add_pd(v1, v2); }
    static register_type load(const value_type* px) { return _mm256_load_pd(px); }
    static register_type loadu(const value_type* px) { return _mm256_loadu_pd(px); }
    static void store(value_type* px, const register_type& v) { _mm256_store_pd(px, v); }
    static void storeu(value_type* px, const register_type& v) { _mm256_storeu_pd(px, v); }
    static value_type resemble(const register_type& x) {
        __declspec(align(32)) double tmp[4];
        _mm256_store_pd(tmp, x);
        return std::accumulate(tmp, tmp + 4, 0.0);  
    }
};

template<typename T>
struct avx {};
template<>
struct avx<float> : public float_avx {};
template<>
struct avx<double> : public double_avx {};

#endif // USE_AVX

template<typename T>
inline bool is_aligned(const typename T::value_type* p) {
    return reinterpret_cast<size_t>(p) % (sizeof(T::unroll_size) * sizeof(typename T::value_type*)) == 0;
}

template<typename T>
inline bool is_aligned(const typename T::value_type* p1, const typename T::value_type* p2) {
    return is_aligned(p1) && is_aligned(p2);
}

// generic dot-product
template<typename T>
inline typename T::value_type dot_product_nonaligned(const typename T::value_type* f1, const typename T::value_type* f2, unsigned int size) {
    typename T::register_type result = {};

    for (unsigned int i = 0; i < size/T::unroll_size; i++) 
        result = T::add(result, T::mul(T::loadu(&f1[i*T::unroll_size]), T::loadu(&f2[i*T::unroll_size])));  

    typename T::value_type sum = T::resemble(result);

    for (unsigned int i = (size/T::unroll_size)*T::unroll_size; i < size; i++)
        sum += f1[i] * f2[i];

    return sum;
}

// generic dot-product(aligned)
template<typename T>
inline typename T::value_type dot_product_aligned(const typename T::value_type* f1, const typename T::value_type* f2, unsigned int size) {
    typename T::register_type result = T::zero();

    assert(is_aligned<T>(f1));
    assert(is_aligned<T>(f2));

    for (unsigned int i = 0; i < size/T::unroll_size; i++) 
        result = T::add(result, T::mul(T::load(&f1[i*T::unroll_size]), T::load(&f2[i*T::unroll_size])));  
    
    typename T::value_type sum = T::resemble(result);

    for (unsigned int i = (size/T::unroll_size)*T::unroll_size; i < size; i++)
        sum += f1[i] * f2[i];

    return sum;
}

template<typename T>
inline void muladd_aligned(const typename T::value_type* src, typename T::value_type c, unsigned int size, typename T::value_type* dst) {
    typename T::register_type factor = T::set1(c);

    for (unsigned int i = 0; i < size/T::unroll_size; i++) {
        T::register_type d = T::load(&dst[i*T::unroll_size]);
        T::register_type s = T::load(&src[i*T::unroll_size]);
        T::store(&dst[i*T::unroll_size], T::add(d, T::mul(s, factor)));
    }

    for (unsigned int i = (size/T::unroll_size)*T::unroll_size; i < size; i++)
        dst[i] += src[i] * c;
}


template<typename T>
inline void muladd_nonaligned(const typename T::value_type* src, typename T::value_type c, unsigned int size, typename T::value_type* dst) {
    typename T::register_type factor = T::set1(c);

    for (unsigned int i = 0; i < size/T::unroll_size; i++) {
        T::register_type d = T::loadu(&dst[i*T::unroll_size]);
        T::register_type s = T::loadu(&src[i*T::unroll_size]);
        T::storeu(&dst[i*T::unroll_size], T::add(d, T::mul(s, factor)));
    }

    for (unsigned int i = (size/T::unroll_size)*T::unroll_size; i < size; i++)
        dst[i] += src[i] * c;
}
} // namespace detail


// dst[i] += c * src[i]
template<typename T>
void muladd(const T* src, T c, unsigned int size, T* dst) {
#if defined(USE_AVX)
    if (is_aligned<detail::avx<T> >(src, dst))
        muladd_aligned<detail::avx<T> >(src, c, size, dst);
    else
        muladd_nonaligned<detail::avx<T> >(src, c, size, dst);
#elif defined(USE_SSE)
    if (is_aligned<detail::sse<T> >(src, dst))
        muladd_aligned<detail::sse<T> >(src, c, size, dst);
    else
        muladd_nonaligned<detail::sse<T> >(src, c, size, dst);
#else
    muladd_aligned<detail::generic<T> >(src, c, size, dst);
#endif
}

// sum(s1[i] * s2[i])
template<typename T>
T dot(const T* s1, const T* s2, unsigned int size) {
#if defined(USE_AVX)
    if (is_aligned<detail::avx<T> >(s1, s2))
        dot_product_aligned<detail::avx<T> >(s1, s2, size);
    else
        dot_product_nonaligned<detail::avx<T> >(s1, s2, size);
#elif defined(USE_SSE)
    if (is_aligned<detail::sse<T> >(s1, s2))
        dot_product_aligned<detail::sse<T> >(s1, s2, size);
    else
        dot_product_nonaligned<detail::sse<T> >(s1, s2, size);
#else
    dot_product_aligned<detail::generic<T> >(s1, s2, size);
#endif
}

} // namespace vectorize
