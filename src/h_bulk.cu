/*******************************************************
Copyright Fujitsu Limited and Hiroshima University 2023
All rights reserved.

This software is the confidential and proprietary 
information of Fujitsu Limited and Hiroshima University.
/*******************************************************/


#include <cstdio>
#include <iostream>
#include <string>
#include <chrono>

#include <cuda.h>

#include "mp.h"
#include "host.h"
#include "constants.h"


void hostBulkEvaluation(Degree* n, double* x, double* boys, 
                        std::string scenario, const TaylorTable& LUT, 
                        int num_inputs, int num_evals)
{
    float mean_ms = 0.0;
    const int num_samples = 10;
    std::chrono::system_clock::time_point start, end;

    size_t boys_size = 0;
    if (scenario == "single") {
        boys_size = sizeof(double) * num_inputs;
        //cudaMallocHost(&boys, boys_size);

        for (int sid = 0; sid < num_samples; ++sid) {
            cudaMemset(boys, 0, boys_size);
            start = std::chrono::system_clock::now();
            hostSingleEvaluation(n, x, boys, LUT, num_inputs);
            end = std::chrono::system_clock::now();
            auto elapsed_time = end - start;
            auto elapsed_ms = std::chrono::duration_cast\
                              <std::chrono::microseconds>\
                              (elapsed_time).count() / 1000.0;
            mean_ms += elapsed_ms;
        }
    }
    else if (scenario == "incremental") {
        boys_size = sizeof(double) * (LUT.n_max + 1) * num_inputs;
        //cudaMallocHost(&boys, boys_size);
        
        for (int sid = 0; sid < num_samples; ++sid) {
            cudaMemset(boys, 0, boys_size);
            start = std::chrono::system_clock::now();
            hostIncrementalEvaluation(n, x, boys, LUT, num_inputs);
            end = std::chrono::system_clock::now();
            auto elapsed_time = end - start;
            auto elapsed_ms = std::chrono::duration_cast\
                              <std::chrono::microseconds>\
                              (elapsed_time).count() / 1000.0;
            mean_ms += elapsed_ms;
        }
    }

    mean_ms /= num_samples;

    printf("---<HOST COMPUTATION TIME>---\n");
    printf("#inputs: %d\n", num_inputs);
    printf("boys evaluation: %.2f [ms]\n", mean_ms);
}


