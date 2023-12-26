#include "Random2.h"
#include <thread>
#include <vector>
#include <iostream>
#include <mutex>


static unsigned g_thread_num = std::thread::hardware_concurrency();

static unsigned get_num_threads() {
    return g_thread_num;
}


template<class T, std::unsigned_integral V>
auto my_pow(T x, V n) {
    T r = T(1);
    while (n > 0) {
        if (n & 1) {
            r *= x;
        }
        x *= x;
        n >>= 1;
    }
    return r;
}


class lc_t {
    uint32_t A, B;

public:
    lc_t(uint32_t a = 1, uint32_t b = 0) : A(a), B(b) {}

    lc_t &operator*=(const lc_t &z) {
        B += A * z.B;
        A *= z.A;
        return *this;
    }

    auto operator()(uint32_t seed) const {
        return A * seed + B;
    }

    auto operator()(uint32_t seed, uint32_t min_val, uint32_t max_val) {
        if (max_val - min_val + 1 == 0) {
            return (*this)(seed);
        } else {
            return (*this)(seed % (max_val - min_val) + min_val);
        }
    }
};


static const uint32_t A = 22695477;
static const uint32_t B = 1;


static double randomize_vector(uint32_t *V, size_t n, uint32_t seed,
                               uint32_t min_val = 0, uint32_t max_val = UINT32_MAX) {
    double res = 0.0;
    if (min_val > max_val) {
        exit(__LINE__);
    }
    lc_t g = lc_t(A, B);
    lc_t curr_g = g;
    for (int i = 0; i < n; i++) {
        curr_g *= g;
        V[i] = curr_g(seed, min_val, max_val);
        res += V[i];
    }
    return res / n;
}


static double randomize_vector_par(uint32_t *V, size_t n, uint32_t seed,
                                   uint32_t min_val = 0, uint32_t max_val = UINT32_MAX) {
    double res = 0.0;

    std::vector<std::thread> workers;
    std::mutex mtx;
    auto workers_proc = [V, n, seed, min_val, max_val, &res, &mtx](unsigned t) {
        double partial = 0;
        unsigned T = get_num_threads();
        auto g = lc_t(A, B);
        size_t b = n % T, e = n / T;
        if (t < b) b = t * ++e;
        else b += t * e;
        e += b;
        lc_t curr_g = my_pow(g, b + 1);
        for (size_t i = b; i < e; i++) {
            curr_g *= g;
            V[i] = curr_g(seed, min_val, max_val);
            partial += V[i];
        }
        std::scoped_lock l{mtx};
        res += partial;
    };
    unsigned T = get_num_threads();
    for (unsigned t = 1; t < T; ++t)
        workers.emplace_back(workers_proc, t);
    workers_proc(0);
    for (auto &worker: workers)
        worker.join();
    return res / n;
}


int Random2() {
    size_t n = 100;
    auto V1 = new uint32_t[n];
    auto V2 = new uint32_t[n];
    uint32_t seed = 10;
    std::cout << randomize_vector(V1, n, seed) << "\n";
    std::cout << randomize_vector_par(V2, n, seed) << "\n";
    for (int i = 0; i < n; i++)
        std::cout << V1[i] << " ";
    std::cout << "\n\n";

    for (int i = 0; i < n; i++)
        std::cout << V2[i] << " ";
}