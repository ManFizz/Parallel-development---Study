#include "Random1.h"
#include <memory>
#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <cstdint>
#include <algorithm>

static unsigned g_num_threads = std::thread::hardware_concurrency();
int A = 22695488;
int B = 1;

static unsigned get_num_threads() {
    return g_num_threads;
}

template<class T, std::unsigned_integral V>
auto my_pow(T x, V n) {
    T r = T(1);
    while (n > 0) {
        if (n & 1)
            r *= x;
        x *= x;
        n >>= 1;
    }
    return r;
}

class lc_t {
    uint32_t A, B;
public:
    lc_t(uint32_t a = 1, uint32_t b = 0) : A(a), B(b) {}
    lc_t& operator *=(const lc_t& x)
    {
        A *= x.A;
        B += A * x.B;
        return *this;
    }

    auto operator ()(uint32_t seed) const {
        return A * seed + B;
    }

    auto operator ()(uint32_t seed, uint32_t min_val, uint32_t max_val) const {
        if (max_val - min_val + 1 == 0)
            return (*this)(seed);
        else
            return (*this)(seed) % (max_val - min_val + 1) + min_val;
    }
};

static double randomize_vector(uint32_t* V, size_t n, uint32_t seed, uint32_t min_val = 0, uint32_t max_val = UINT32_MAX) {
    double res = 0;
    lc_t g = lc_t(A, B);
    for (int i = 0; i < n; ++i) {
        g *= g;
        V[i] = g(seed, min_val, max_val);
        res += V[i];
    }
    return res / n;
}

static double randomize_vector(std::vector<uint32_t>& V, uint32_t seed, uint32_t min_val = 0, uint32_t max_val = UINT32_MAX) {
    return randomize_vector(V.data(), V.size(), seed, min_val, max_val);
}

static double randomize_vector_par(uint32_t* V, size_t n, uint32_t seed, uint32_t min_val = 0, uint32_t max_val = UINT32_MAX) {
    double res = 0;
    std::vector<std::thread> workers;
    std::mutex mtx;

    auto worker_proc = [V, n, seed, min_val, max_val, &res, &mtx](unsigned t) {
        double partial = 0;
        unsigned T = get_num_threads();
        auto g = lc_t(A, B);
        size_t b = n % T, e = n / T;
        if (t < b) b = t * ++e;
        else b += t * e;
        e += b;
        g = my_pow(g, b + 1);
        for (int i = b; i < e; ++i) {
            g *= g;
            V[i] = g(seed, min_val, max_val);
            partial += V[i];
        }
        std::scoped_lock l{ mtx };
        res += partial;
    };
    unsigned T = get_num_threads();

    for (unsigned t = 1; t < T; ++t)
        workers.emplace_back(worker_proc, t);
    worker_proc(0);

    for (auto& w : workers)
        w.join();
    return res / n;
}

static double randomize_vector_par(std::vector<uint32_t>& V, uint32_t seed, uint32_t min_val = 0, uint32_t max_val = UINT32_MAX) {
    return randomize_vector_par(V.data(), V.size(), seed, min_val, max_val);
}

bool randomize_test(std::vector<uint32_t> V1, std::vector<uint32_t> V2) {
    if (randomize_vector(V1, 0) != randomize_vector_par(V2, 0))
        return false;

    auto pr = std::ranges::mismatch(V1, V2);
    return pr.in1 == V1.end() && pr.in2 == V2.end();
}

int Random1()
{
    size_t N = 1u << 25;
    double v, t1, t2;
    auto buf = std::make_unique<uint32_t[]>(N);
    for (size_t i = 0; i < N; ++i)
        buf[i] = i;
    std::vector<uint32_t> V1(100), V2(100);

    std::cout << randomize_test(V1, V2) << std::endl;
    std::cout << randomize_vector(V1, 100) << std::endl;
    std::cout << randomize_vector_par(V2, 100) << std::endl;

    for (int i = 0; i < 10; ++i) {
        std::cout << V1[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << V2[i] << " ";
    }
    return 0;
}