#include <cstdlib> 
#include <vector>
#include <numeric>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <string>
#include <random>
#include <cmath>
#include <fstream>
#include <cstring>
#include <cassert>
#include <omp.h>

#include "mkl.h"
#include "mkl_types.h"
#include "mkl_cblas.h"

const int WARMUP = 100;
const int RUNS = 1000;
const float ERR = 0.05;
double FLOPs;

void matmul(const float * const A, const float * const B, float * const C, const int M, const int K, const int N);
void compare(const float * const A, const float * const B, const int M, const int N);
void initInputs(float * const A, float * const B, const int M, const int K, const int N);
void printResults(const std::string& name, const std::vector<double>& results, const double FLOPs);
double onemkl(const float *A, const float *B, float *C, const int M, const int K, const int N);

int main(int argc, char* argv[])
{
    if (argc != 5) {
        std::cerr << "Invalid Usage!\n";
        std::cout << "Usage: ./onemkl_sgemm M K N nthread\n";
        exit(1);
    }

    int M = std::stoi(argv[1]);
    int K = std::stoi(argv[2]);
    int N = std::stoi(argv[3]);
    int nthread = std::stoi(argv[4]);
    assert(M > 0 && K > 0 && N > 0 && nthread > 0);
    FLOPs = 2.0 * M * K * N;

    std::cout << "M=" << M << " K=" << K << " N=" << N << " nthread=" << nthread << "\n";
    omp_set_num_threads(nthread);

    float *A = (float*)malloc(sizeof(float) * M * K);
    float *B = (float*)malloc(sizeof(float) * K * N);
    initInputs(A, B, M, K, N);
    float *C = (float*)malloc(sizeof(float) * M * N);
    std::memset(C, 0, sizeof(float) * M * N);
    float *D = (float*)malloc(sizeof(float) * M * N);
    std::memset(D, 0, sizeof(float) * M * N);

    /*
    // check correctness first
    matmul(A, B, D, M, K, N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);
    compare(C, D, M, N);
    */
    std::vector<double> timings;
    mkl_set_num_threads(nthread);
    for (int i = 0; i < WARMUP; ++i) cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);
    for (int i = 0; i < RUNS; ++i) timings.push_back(onemkl(A, B, C, M, K, N));
    std::sort(timings.begin(), timings.end());
    printResults("ThreadNum=" + std::to_string(nthread), timings, FLOPs);
    
    // check correctness last
    matmul(A, B, D, M, K, N);
    compare(C, D, M, N);

    free(A);
    free(B);
    free(C);
    free(D);
    return 0;
}


double onemkl(const float *A, const float *B, float *C, const int M, const int K, const int N)
{
    auto start = std::chrono::high_resolution_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
}


void initInputs(float * const A, float * const B, const int M, const int K, const int N)
{
    std::mt19937 engine{137};
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    #pragma omp parallel for
    for (int i = 0; i < M * K; ++i) A[i] = dist(engine);
    #pragma omp parallel for
    for (int i = 0; i < K * N; ++i) B[i] = dist(engine);
}

void matmul(const float * const A, const float * const B, float * const C, const int M, const int K, const int N)
{
    #pragma omp parallel for
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

void compare(const float * const A, const float * const B, const int M, const int N)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (std::abs(A[i * N + j] - B[i * N + j]) > ERR) {
                std::cerr << "Correctness check failed!\n" << "M=" << M << " N=" << N << std::endl;
                std::cerr << "i=" << i << ", j=" << j << " A=" << A[i * N + j] << ", B=" << B[i * N + j] << std::endl;
                exit(1);
            }
        }
    }
    std::cout << "Correctness check passed!\n";
}

void printResults(const std::string& name, const std::vector<double>& results, const double FLOPs)
{
    //double median = results[results.size()/2];
    //std::cout << name << "\tPerformance=" << std::lround(FLOPs/1.0e9/median) << " GFLOPs\n\n";
    
    double total = std::accumulate(results.begin(), results.end(), 0.0);
    double avg = total/results.size();
    double median = results[results.size()/2];
    double min = results[0];
    double dev = 0.0;

    for (const auto re : results)
        dev += (re - avg) * (re - avg);
    dev /= results.size();

    std::cout << "=== " << name << " ===\n";
    std::cout << "Took " << total << " seconds for " << RUNS << " runs.\t" << WARMUP << " warmups.\n";
    std::cout << "Avg\t" << std::lround(FLOPs/1.0e9/avg) << " GFLOPS\n";
    std::cout << "Med\t" << std::lround(FLOPs/1.0e9/median) << " GFLOPS\n";
    std::cout << "Max\t" << std::lround(FLOPs/1.0e9/min) << " GFLOPS\n";
    std::cout << "Dev\t" << dev << "\n\n";
    /*
    std::cout << avg << " Avg.\t(" << std::lround(FLOPs/1.0e9/avg) << " GFLOPS)\n";
    std::cout << median << " Med.\t(" << std::lround(FLOPs/1.0e9/median) << " GFLOPS)\n";
    std::cout << min << " Max.\t(" << std::lround(FLOPs/1.0e9/min) << " GFLOPS)\n";
    std::cout << dev << " Dev.\t(" << dev << ")\n\n";
    */
}