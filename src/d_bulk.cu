#include <cstdio>
#include <string>
#include <map>

#include <cuda.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "mp.h"
#include "device.cuh"
#include "constants.h"


void deviceBulkEvaluation(Degree* h_n, double* h_x, double* h_boys, 
                          std::string scenario, const TaylorTable& h_LUT, 
                          int num_inputs, int num_evals)
{
    Degree* d_n; 
    double* d_x;
    double* d_x_sorted;
    double* d_boys_grid;
    Sortkey* d_key; 
    Sortkey* d_key_sorted;
    double* d_boys; 
    int* d_counter;

    const size_t n_size = sizeof(Degree) * num_inputs;
    const size_t x_size = sizeof(double) * num_inputs;
    size_t boys_size = 0;
    if (scenario == "single") {
        boys_size = sizeof(double) * num_inputs;
    }
    else if (scenario == "incremental") {
        boys_size = sizeof(double) * (h_LUT.n_max + 1) * num_inputs;
    }
    const size_t key_size = sizeof(Sortkey) * num_inputs;
    const size_t counter_size = sizeof(int);

    cudaMalloc(&d_n, n_size);
    cudaMalloc(&d_x, x_size);
    cudaMalloc(&d_x_sorted, x_size);
    cudaMalloc(&d_boys_grid, h_LUT.table_size);
    cudaMalloc(&d_key, key_size);
    cudaMalloc(&d_key_sorted, key_size);
    cudaMalloc(&d_boys, boys_size);
    cudaMalloc(&d_counter, counter_size);

    cudaMemcpy(d_n, h_n, n_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_boys_grid, h_LUT.boys_grid, h_LUT.table_size, cudaMemcpyHostToDevice);

    cudaFuncSetCacheConfig(&singleBulkEvaluation, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(&incrementalBulkEvaluation, cudaFuncCachePreferL1);

    // allocate temporary memory for sorting
    void *d_tmp = NULL;
    size_t d_tmp_size = 0;
    cub::DeviceRadixSort::SortPairsDescending(d_tmp, d_tmp_size, 
                                              d_key, d_key_sorted, 
                                              d_x, d_x_sorted, num_inputs);
    cudaMalloc(&d_tmp, d_tmp_size);

    const int num_samples = 10;
    std::map<std::string, float> kernel_time;
    kernel_time["key"] = 0.0f;
    kernel_time["sort"] = 0.0f;
    kernel_time["boys"] = 0.0f;

    const int threadsPerWarp = 32;
    const int threadsPerBlock = 128;
    const int warpsPerBlock = threadsPerBlock / threadsPerWarp;
    const int num_blocks = num_inputs / threadsPerBlock;
    dim3 blocks(num_blocks);
    dim3 threads(threadsPerWarp, warpsPerBlock);

    const int tpb = 1024;
    const int nb = num_inputs / tpb;

    cudaEvent_t begin, end;
    float elapsed_time = 0.0f;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    for (int s = 0; s < num_samples; ++s) {
        cudaMemset(d_boys, 0, boys_size);
        cudaMemset(d_counter, 0, counter_size);

        //*
        if (scenario == "single") {
            cudaEventRecord(begin);
            generateKeySingle<<<nb, tpb>>>(d_n, d_x, d_key);
            cudaEventRecord(end);
        }
        else if (scenario == "incremental") {
            cudaEventRecord(begin);
            generateKeyIncremental<<<nb, tpb>>>(d_n, d_x, d_key);
            cudaEventRecord(end);
        }
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, begin, end);
        kernel_time["key"] += elapsed_time;
        /**/

        cudaDeviceSynchronize();

        /*
        cudaEventRecord(begin);
        cub::DeviceRadixSort::SortPairsDescending(d_tmp, d_tmp_size, 
                                                  d_key, d_key_sorted, 
                                                  d_x, d_x_sorted, num_inputs);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, begin, end);
        kernel_time["sort"] += elapsed_time;
        /**/

        //*
        if (scenario == "single") {
            cudaEventRecord(begin);
            singleBulkEvaluation<<<blocks, threads>>>(d_key, d_x, d_boys, d_boys_grid, d_counter);
            // singleBulkEvaluation<<<blocks, threads>>>(d_key_sorted, d_x_sorted, d_boys, d_boys_grid, d_counter);
            cudaEventRecord(end);
        }
        else if (scenario == "incremental") {
            cudaEventRecord(begin);
            incrementalBulkEvaluation<<<blocks, threads>>>(d_key, d_x, d_boys, d_boys_grid, d_counter, num_inputs);
            // incrementalBulkEvaluation<<<blocks, threads>>>(d_key_sorted, d_x_sorted, d_boys, d_boys_grid, d_counter, num_inputs);
            cudaEventRecord(end);
        }
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, begin, end);
        kernel_time["boys"] += elapsed_time;
        /**/
    }
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaDeviceSynchronize();

    cudaMemcpy(h_boys, d_boys, boys_size, cudaMemcpyDeviceToHost);

    kernel_time["key"] /= num_samples;
    kernel_time["sort"] /= num_samples;
    kernel_time["boys"] /= num_samples;
    printf("---<DEVICE COMPUTATION TIME>---\n");
    printf("key generation: %.2f [ms]\n", kernel_time["key"]);
    printf("input sorting: %.2f [ms]\n", kernel_time["sort"]);
    printf("boys evaluation: %.2f [ms]\n", kernel_time["boys"]);

    cudaFree(d_n);
    cudaFree(d_x);
    cudaFree(d_x_sorted);
    cudaFree(d_key);
    cudaFree(d_key_sorted);
    cudaFree(d_boys_grid);
    cudaFree(d_boys);
    cudaFree(d_counter);
    cudaFree(d_tmp);
}


