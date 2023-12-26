#include "Overage1.h"
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <condition_variable>
#include <pthread.h>

extern "C"
{
    double average_rr_pthread(const double *v, size_t n);
    double average_pthread_local(const double *v, size_t n);
    void pthread_init();
    void pthread_destroy();
}
static unsigned g_num_threads = std::thread::hardware_concurrency();
static unsigned get_num_thread() {
    return g_num_threads;
}

void set_num_threads(unsigned T);

struct partial_sum_t
{
    alignas(64) double value;
};

typedef struct profiling_results_t
{
    double result, time, speedup, efficiency;
    unsigned T;
} profiling_results_t;

template <class F>
auto run_experiment(F func, const double *v, size_t n)
requires std::is_invocable_r_v<double, F, const double *, size_t>
{
    std::vector<profiling_results_t> res_table;
    auto Tmax = get_num_thread(); 
    for (unsigned int T = 1; T <= Tmax; ++T)
    {
        using namespace std::chrono;
        res_table.emplace_back();
        auto& rr = res_table.back();
        set_num_threads(T);
        auto t1 = steady_clock::now();
        rr.result = func(v, n);
        auto t2 = steady_clock::now();
        rr.time = duration_cast<milliseconds>(t2 - t1).count();
        rr.speedup = res_table.front().time / rr.time;
        rr.efficiency = rr.time / T;
        rr.T = T;
    }
    return res_table;
}

typedef double (*avg_t)(const double *, size_t);

double get_omp_time(avg_t func, const double *v, size_t n)
{
    auto t1 = omp_get_wtime();
    func(v, n);
    auto t2 = omp_get_wtime();
    return t2 - t1;
}

template <class F>
    requires std::invocable<double, F, const double *, size_t>
auto cpp_get_time(F f, const double *v, size_t n)
{
    using namespace std::chrono;
    auto t1 = steady_clock::now();
    f(v, n);
    auto t2 = steady_clock::now();
    return duration_cast<milliseconds>(t2 - t1).count();
}

double average(const double *v, size_t n)
{
    double res = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        res += v[i];
    }
    return res / n;
}

double average_reduce(const double *v, size_t n)
{
    double res = 0.0;
#pragma omp parallel for reduction(+ : res)
    for (size_t i = 0; i < n; ++i)
    {
        res += v[i];
    }
    return res / n;
}

double average_rr(const double *v, size_t n) // Roll Round
{
    double res = 0.0;
#pragma omp parallel
    {
        unsigned t = omp_get_thread_num();
        unsigned T = omp_get_num_threads();
        for (size_t i = t; i < n; i += T)
        {
            res += v[i]; // Гонка
        }
    }
    return res / n;
}

double average_omp(const double *v, size_t n)
{
    double res = 0.0, *partial_sums;
    unsigned T;
#pragma omp parallel shared(T)
    {
        unsigned t = omp_get_thread_num();
#pragma omp single
        {
            T = omp_get_num_threads();
            partial_sums = (double *)calloc(T, sizeof(v[0]));
        }
        for (size_t i = t; i < n; i += T)
        {
            partial_sums[t] += v[i];
        }
    }
    for (size_t i = 1; i < omp_get_num_procs(); ++i)
    {
        partial_sums[0] += partial_sums[i];
    }
    res = partial_sums[0] / n;
    free(partial_sums);
    return res;
}

double average_omp_align(const double *v, size_t n)
{
    double res = 0.0;
    partial_sum_t *partial_sums;
    unsigned T;
#pragma omp parallel shared(T)
    {
        unsigned t = omp_get_thread_num();
#pragma omp single
        {
            T = omp_get_num_threads();
            partial_sums = (partial_sum_t *)calloc(T, sizeof(partial_sum_t));
        }
        for (size_t i = t; i < n; i += T)
        {
            partial_sums[t].value += v[i];
        }
    }
    for (size_t i = 1; i < T; ++i)
    {
        partial_sums[0].value += partial_sums[i].value;
    }
    res = partial_sums[0].value / n;
    free(partial_sums);
    return res;
}

double average_omp_mtx(const double *v, size_t n)
{
    double res = 0.0;
#pragma omp parallel
    {
        unsigned int t = omp_get_thread_num();
        unsigned int T = omp_get_num_threads();
        for (size_t i = t; i < n; i += T)
        {
#pragma omp critial
            {
                res += v[i];
            }
        }
    }
    return res / n;
}

double average_omp_mtx_opt(const double *v, size_t n)
{
    double res = 0.0;
#pragma omp parallel
    {
        double partial = 0.0;
        unsigned int t = omp_get_thread_num();
        unsigned int T = omp_get_num_threads();
        for (size_t i = t; i < n; i += T)
        {

            partial += v[i];
        }
#pragma omp critical
        {
            res += partial;
        }
    }
    return res / n;
}

double average_cpp_mtx(const double *v, size_t n)
{
    double res = 0.0;
    unsigned T = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    std::mutex mtx;
    auto worker_proc = [&mtx, T, n, &res, v](unsigned t)
    {
        double partial_result = 0.0;
        for (std::size_t i = t; i < n; i += T)
        {
            partial_result += v[i];
        }
        std::scoped_lock l{mtx};
        res += partial_result;
    };
    for (unsigned t = 1; t < T; ++t)
    {
        workers.emplace_back(worker_proc, t);
    }
    worker_proc(0);
    for (auto &w : workers)
    {
        w.join();
    }
    return res / n;
}

double average_cpp_partial_align(const double *v, size_t n)
{
    double res = 0.0;
    unsigned T = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    partial_sum_t *partial_sums = (partial_sum_t *)calloc(T, sizeof(partial_sum_t));
    auto worker_proc = [v, n, T, &res, partial_sums](size_t t)
    {
        for (size_t i = t; i < n; i += T)
        {
            partial_sums[t].value += v[i];
        }
    };
    for (unsigned t = 1; t < T; ++t)
    {
        workers.emplace_back(worker_proc, t);
    }
    worker_proc(0);
    for (auto &w : workers)
    {
        w.join();
    }
    for (size_t i = 1; i < T; ++i)
    {
        partial_sums[0].value += partial_sums[i].value;
    }
    res = partial_sums[0].value / n;
    free(partial_sums);
    return res;
}

double average_mtx_local(const double *v, size_t n)
{
    double res = 0.0;
    unsigned T = get_num_thread();
    std::mutex mtx;
    size_t e = n / T;
    size_t b = n % T;
    std::vector<std::thread> workers;
    auto worker_proc = [v, n, T, &res, &mtx](size_t t, size_t e, size_t b)
    {
        double local = 0.0;
        if (t < b)
        {
            b = t * ++e;
        }
        else
        {
            b += t * e;
        }
        e += b;
        for (size_t i = b; i < e; ++i)
        {
            local += v[i];
        }
        std::scoped_lock l{mtx};
        res += local;
    };
    for (unsigned t = 1; t < T; ++t)
    {
        workers.emplace_back(worker_proc, t, e, b);
    }
    worker_proc(0, e, b);
    for (auto &w : workers)
    {
        w.join();
    }
    return res / n;
}

class barrier {
    unsigned lock_id = 0;
    unsigned T;
    unsigned Tmax;
    std::mutex mtx;
    std::condition_variable cv;
public:
    barrier(unsigned threads) : T(threads), Tmax(threads) {}
    void arrive_and_wait() {
        std::unique_lock l{mtx};
        if (--T) {
            unsigned my_lock_id = lock_id;
            while (my_lock_id == lock_id)
                cv.wait(l);
        }
        else {
            ++lock_id;
            cv.notify_all();
            T = Tmax;
        }
    }
};

double average_cpp_reduction() {
    unsigned T = std::thread::hardware_concurrency();
    std::vector<double> partial_results(T);
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution dist(-5.0, 5.0);
    for (auto& elem : partial_results) {
        elem = dist(mt);
    }
    auto backup = partial_results;

    barrier bar(T);

    // auto work = [&]

}

int Overage1()
{
    pthread_init();
    size_t N = 1u << 25;
    double v, t1, t2;
    char pattern[] = "%f, %f %s\n";

    auto buf = std::make_unique<double[]>(N);
    for (size_t i = 0; i < N; ++i)
        buf[i] = i;


    t1 = omp_get_wtime();
    v = average(buf.get(), N);
    t2 = omp_get_wtime();
    printf(pattern, v, t2 - t1, "sequential");

    t1 = omp_get_wtime();
    v = average_reduce(buf.get(), N);
    t2 = omp_get_wtime();
    printf(pattern, v, t2 - t1, "reduce");

    t1 = omp_get_wtime();
    v = average_rr(buf.get(), N);
    t2 = omp_get_wtime();
    printf(pattern, v, t2 - t1, "rr");

    t1 = omp_get_wtime();
    v = average_omp(buf.get(), N);
    t2 = omp_get_wtime();
    printf(pattern, v, t2 - t1, "omp");

    t1 = omp_get_wtime();
    v = average_omp_align(buf.get(), N);
    t2 = omp_get_wtime();
    printf(pattern, v, t2 - t1, "omp_align");

    t1 = omp_get_wtime();
    v = average_rr_pthread(buf.get(), N);
    t2 = omp_get_wtime();
    printf(pattern, v, t2 - t1, "pthread_rr");

    t1 = omp_get_wtime();
    v = average_omp_mtx(buf.get(), N);
    t2 = omp_get_wtime();
    printf(pattern, v, t2 - t1, "omp_mtx");

    t1 = omp_get_wtime();
    v = average_omp_mtx_opt(buf.get(), N);
    t2 = omp_get_wtime();
    printf(pattern, v, t2 - t1, "omp_mtx_opt");

    t1 = omp_get_wtime();
    v = average_cpp_mtx(buf.get(), N);
    t2 = omp_get_wtime();
    printf(pattern, v, t2 - t1, "cpp_mtx");

    t1 = omp_get_wtime();
    v = average_cpp_partial_align(buf.get(), N);
    t2 = omp_get_wtime();
    printf(pattern, v, t2 - t1, "cpp_partial_align");

    t1 = omp_get_wtime();
    v = average_mtx_local(buf.get(), N);
    t2 = omp_get_wtime();
    printf(pattern, v, t2 - t1, "cpp_mtx_local");

    t1 = omp_get_wtime();
    v = average_pthread_local(buf.get(), N);
    t2 = omp_get_wtime();
    printf(pattern, v, t2 - t1, "pthread_local");

    auto a = run_experiment(average_mtx_local, buf.get(), N);

    for (auto& i: a)
    {
        std::cout << "result " << i.result << '\n';
        std::cout << "time " << i.time << '\n';
        std::cout << "speedup " << i.speedup << '\n';
        std::cout << "efficiency " << i.efficiency << '\n';
        std::cout << "T " << i.T << '\n';
        std::cout << '\n';
    }

    pthread_destroy();
    return 0;
}
