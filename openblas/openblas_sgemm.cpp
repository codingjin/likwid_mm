#include <cblas.h>
#include <stdlib.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <random>
#include <iostream>

const int RUNS = 1000;
void initInputs(float * const A, float * const B, const int M, const int K, const int N);

int main(int argc, char* argv[])
{
    if (argc != 5) {
        std::cerr << "Invalid Usage!\n";
        std::cout << "Usage: ./openblas_sgemm M K N nthread\n";
        exit(1);
    }

    int M = std::stoi(argv[1]);
    int K = std::stoi(argv[2]);
    int N = std::stoi(argv[3]);
    int nthread = std::stoi(argv[4]);
    openblas_set_num_threads(nthread);

    float *A = (float*)malloc(sizeof(float) * M * K);
    float *B = (float*)malloc(sizeof(float) * K * N);
    initInputs(A, B, M, K, N);
    float *C = (float*)malloc(sizeof(float) * M * N);
    std::memset(C, 0, sizeof(float) * M * N);

    for (int i = 0; i < RUNS; ++i) 
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);

    //free(A);
    //free(B);
    //free(C);
    return 0;
}

void initInputs(float * const A, float * const B, const int M, const int K, const int N)
{
    std::mt19937 engine{137};
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < M * K; ++i) A[i] = dist(engine);
    for (int i = 0; i < K * N; ++i) B[i] = dist(engine);
}

