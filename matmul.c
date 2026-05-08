/*
 * Prakitikum 1 – Matrix Multiplication on CPU
 * ============================================
 * AI Accelerators (AIA) – Lab Assignment
 *
 * Your task is to progressively optimize this naive C implementation
 * of matrix multiplication (C = A * B) through the steps below.
 * Read README.md carefully before you start!
 *
 * Build:  make
 * Run:    ./matmul <size>      (e.g. ./matmul 512)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>
#include <windows.h>

#define num_threads 4
const int num_iterations = 4;
#define JB 64 // Tile size divides matrix size
#define COMPILER_BARRIER(x) (*(volatile float *)&(x) = (x))

// ============================================================================
// IMPLEMENTATION 1: NAIVE MATRIX MULTIPLICATION
// ============================================================================
void matmul_naive(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// IMPLEMENTATION 2: -03 -ffmastmath, does loop unrolling and vectorization, reorder k,j
// ============================================================================
void matmul_looporder(const float *A, const float *B, float *C, int M, int N, int K)
{
    /* TODO: implement your best loop ordering here, replace below*/
    memset(C, 0, M * N * sizeof(float));
    // i-j-k:
    // for (int i = 0; i < M; i++)
    // {
    //     for (int k = 0; k < K; k++)
    //     {
    //         float r = A[i * K + k];
    //         for (int j = 0; j < N; j++)
    //         {
    //             C[i * N + j] += r * B[k * N + j];
    //         }
    //     }
    // }
    // // j-k-i:
    // for (int j = 0; j < N; j++)
    // {
    //     for (int k = 0; k < K; k++)
    //     {
    //         float r = B[k * N + j];
    //         for (int i = 0; i < M; i++)
    //         {
    //             C[i * N + j] += r * A[i * K + k];
    //         }
    //     }
    // }
    // k-i-j (best loop)
    for (int k = 0; k < K; k++)
    {
        for (int i = 0; i < M; i++)
        {
            float r = A[i * K + k];
            for (int j = 0; j < N; j++)
            {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
}

// ============================================================================
// IMPLEMENTATION 3: Tiling
// ============================================================================

void matmul_looptiling(const float *A, const float *B, float *C, int M, int N, int K)
{
    int T = 256;
    memset(C, 0, M * N * sizeof(float));

    for (int tileRow = 0; tileRow < M; tileRow += T)
    {
        for (int tileK = 0; tileK < K; tileK += T)
        {
            for (int tileCol = 0; tileCol < N; tileCol += T)
            {
                for (int i = tileRow; i < tileRow + T; i++)
                {
                    for (int k = tileK; k < tileK + T; k++)
                    {
                        float r = A[i * K + k];
                        for (int j = tileCol; j < tileCol + T; j++)
                        {
                            C[i * N + j] += r * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

typedef struct
{
    const float *A, *B;
    float *C;
    int M, N, K;
    int row_start, row_end;
} ThreadArgs;

DWORD WINAPI thread_func(LPVOID arg)
{
    ThreadArgs *a = (ThreadArgs *)arg;
    for (int i = a->row_start; i < a->row_end; i++)
    {
        for (int k = 0; k < a->K; k++)
        {
            float r = a->A[i * a->K + k];
            for (int j = 0; j < a->N; j++)
                a->C[i * a->N + j] += r * a->B[k * a->N + j];
        }
    }
    return 0;
}

// ============================================================================
// IMPLEMENTATION 4: Multithreading
// ============================================================================
void matmul_parallel_ikj(const float *A, const float *B, float *C, int M, int N, int K)
{
    memset(C, 0, M * N * sizeof(float));

    // for small matrices thread overhead costs more than the gain
    if (M < 256)
    {
        for (int i = 0; i < M; i++)
        {
            for (int k = 0; k < K; k++)
            {
                float r = A[i * K + k];
                for (int j = 0; j < N; j++)
                    C[i * N + j] += r * B[k * N + j];
            }
        }
        return;
    }

    HANDLE threads[num_threads];
    ThreadArgs args[num_threads];
    int rows_per_thread = M / num_threads;

    for (int t = 0; t < num_threads; t++)
    {
        args[t].A = A;
        args[t].B = B;
        args[t].C = C;
        args[t].M = M;
        args[t].N = N;
        args[t].K = K;
        args[t].row_start = t * rows_per_thread;
        args[t].row_end = (t == num_threads - 1) ? M : (t + 1) * rows_per_thread;
        threads[t] = CreateThread(NULL, 0, thread_func, &args[t], 0, NULL);
    }
    WaitForMultipleObjects(num_threads, threads, TRUE, INFINITE);
    for (int t = 0; t < num_threads; t++)
        CloseHandle(threads[t]);
}

// ============================================================================
// Utility functions: Init Matrix, Benchmarking, Calculate Gflops
// ============================================================================
void initialize_matrix(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        matrix[i] = rand() % 100;
    }
}

// Adjustment for windows to make sure get_time_ms() works

#ifdef _WIN32
#include <windows.h>

static double get_time_ms(void)
{
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);
    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / freq.QuadPart * 1000.0;
}

#else
#include <time.h>

static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

#endif

double calculate_gflops(int M, int N, int K, double total_time)
{
    double flops = 2.0 * M * N * K;
    double gflops = (flops / ((total_time) / 1000.0)) / 1e9;
    return gflops;
}

int verify_result(const float *C_ref, const float *C_test, int M, int N, float tolerance)
{
    for (int i = 0; i < M * N; i++)
    {
        if (fabs(C_ref[i] - C_test[i]) > tolerance)
        {
            printf("Mismatch at index %d: ref=%f, test=%f\n", i, C_ref[i], C_test[i]);
            return 0;
        }
    }
    return 1;
}

typedef void (*matmul_fn)(const float *A, const float *B, float *C, int M, int N, int K);

float benchmark(matmul_fn matmul, const float *A, const float *B, float *C, int M, int N, int K)
{
    matmul(A, B, C, M, N, K); // Warmup
    double total_time = 0.0;
    for (int i = 0; i < num_iterations; i++)
    {
        double start = get_time_ms();
        matmul(A, B, C, M, N, K);
        COMPILER_BARRIER(C[0]); // changes to make it suitable for windows implementation.
        double end = get_time_ms();
        total_time += end - start;
    }

    return total_time / num_iterations;
}

// ============================================================================
// Main: Verify results and performance benchmark
// ============================================================================
int main(int argc, char *argv[])
{
    srand(42);
    printf("MatMul Benchmark: Square Matrix\n");

    int sizes[] = {2048, 1024, 512, 256, 128, 64};
    int n = sizeof(sizes) / sizeof(sizes[0]);

    printf("%-8s %-15s %-15s %-15s %-15s\n", "Size", "Naive", "Reordered", "Tiled", "Parallel");
    printf("%-8s %-15s %-15s %-15s %-15s\n", "----", "-----", "---------", "-----", "--------");

    for (int i = 0; i < n; i++)
    {
        int M = sizes[i], N = M, K = M;

        float *A = (float *)malloc(M * K * sizeof(float));
        float *B = (float *)malloc(K * N * sizeof(float));
        float *C = (float *)malloc(M * N * sizeof(float));

        initialize_matrix(A, M, K);
        initialize_matrix(B, K, N);

        // --- 1. Naive ---
        memset(C, 0, M * N * sizeof(float));

        float t_naive = benchmark(matmul_naive, A, B, C, M, N, K);
        double g_naive = calculate_gflops(M, N, K, t_naive);

        // --- 2. Tiled ---
        memset(C, 0, M * N * sizeof(float));

        float t_blocking = benchmark(matmul_looptiling, A, B, C, M, N, K);
        double g_blocking = calculate_gflops(M, N, K, t_blocking);

        // --- 3. Reordered ---
        memset(C, 0, M * N * sizeof(float));

        float t_reorder = benchmark(matmul_looporder, A, B, C, M, N, K);
        double g_reorder = calculate_gflops(M, N, K, t_reorder);

        // --- 4. Parallel ---
        memset(C, 0, M * N * sizeof(float));

        float t_parallel = benchmark(matmul_parallel_ikj, A, B, C, M, N, K);
        double g_parallel = calculate_gflops(M, N, K, t_parallel);

        printf("%d\t%.2f GFLOPS\t%.2f GFLOPS\t%.2f GFLOPS\t%.2f GFLOPS\n", M, g_naive, g_reorder, g_blocking, g_parallel);

        free(A);
        free(B);
        free(C);
    }

    return 0;
}