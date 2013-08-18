#define USE_SSE
#define USE_AVX
#include "product.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <boost/timer.hpp>

using namespace vectorize;
using namespace vectorize::detail;

template<typename Func>
std::pair<double, double> benchmark(Func f) {
    boost::timer t;
    decltype(f()) sum = 0.0;
    for (int i = 0; i < 10000; i++)
        sum += f();
    double time = t.elapsed();
    return std::make_pair((double)sum, time);
}

std::ostream& operator << (std::ostream& os, const std::pair<double, double>& x) {
    os << x.first << "," << x.second;
    return os;
}


template<typename T>
struct aligned_vec {
    aligned_vec(size_t size) : size_(size), p_(size + 8) {
        p_aligned_ = &p_[0];
        while(reinterpret_cast<long>(p_aligned_) % 32) ++p_aligned_;
    }
    aligned_vec(const aligned_vec& rhs) : size_(rhs.size_), p_(rhs.p_) {
        p_aligned_ = &p_[0];
        while(reinterpret_cast<long>(p_aligned_) % 32) ++p_aligned_;
    }
    T& operator [] (size_t index) { return p_aligned_[index]; }
    T* get() { return p_aligned_; }
    const T* get() const { return p_aligned_; }
    aligned_vec<T>& rand_fill() {
        std::mt19937 rng;
        std::uniform_real_distribution<> dst(-1, 1); 
        std::generate(p_aligned_, p_aligned_ + size_, [&rng, &dst](){ return dst(rng); });
        return *this;
    }
private:
    size_t size_;
    std::vector<T> p_;
    T* p_aligned_;
};

template<typename T>
void check_dot() {
    std::mt19937 rng;
    std::uniform_real_distribution<> dst(-1, 1); 
    int len = 1000;

    aligned_vec<T> p1(len);
    aligned_vec<T> p2(len);

    T *vec1 = p1.rand_fill().get();
    T *vec2 = p2.rand_fill().get();

    auto t0 = benchmark([&]{return dot_product_aligned<generic<T> >(vec1, vec2, len);});
    auto t1 = benchmark([&]{return dot_product_aligned<sse<T> >(vec1, vec2, len);});
    auto t2 = benchmark([&]{return dot_product_nonaligned<sse<T> >(vec1, vec2, len);});
    auto t3 = benchmark([&]{return dot_product_aligned<avx<T> >(vec1, vec2, len);});
    auto t4 = benchmark([&]{return dot_product_nonaligned<avx<T> >(vec1, vec2, len);});

    std::cout << "Non-Opt:"       << t0 << "\n"
              << "SSE(align)"     << t1 << "\n"
              << "SSE(non-align)" << t2 << "\n"
              << "AVX(align)"     << t3 << "\n"
              << "AVX(non-align)" << t4 << "\n";
}

template<typename T>
void check_muladd()
{
    int len = 1000;

    aligned_vec<T> src(len);
    std::vector<aligned_vec<T> > dst(5, len);

    src.rand_fill();
    for (auto& d : dst) d.rand_fill();

    auto t0 = benchmark([&] { muladd_aligned<generic<T> >(src.get(), 0.3, len, dst[0].get()); return dst[0][0]; });
    auto t1 = benchmark([&] { muladd_aligned<sse<T> >(src.get(), 0.3, len, dst[1].get()); return dst[1][0]; });
    auto t2 = benchmark([&] { muladd_nonaligned<sse<T> >(src.get(), 0.3, len, dst[2].get()); return dst[2][0]; });
    auto t3 = benchmark([&] { muladd_aligned<avx<T> >(src.get(), 0.3, len, dst[3].get()); return dst[3][0]; });
    auto t4 = benchmark([&] { muladd_nonaligned<avx<T> >(src.get(), 0.3, len, dst[4].get()); return dst[4][0]; });

    std::cout << "Non-Opt:"       << t0 << "\n"
              << "SSE(align)"     << t1 << "\n"
              << "SSE(non-align)" << t2 << "\n"
              << "AVX(align)"     << t3 << "\n"
              << "AVX(non-align)" << t4 << "\n";
}

int main(void) {
    check_dot<float>();
    check_dot<double>();
    check_muladd<float>();
    check_muladd<double>();
}