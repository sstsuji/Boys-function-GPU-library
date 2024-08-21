/*******************************************************
Copyright Fujitsu Limited and Hiroshima University 2023
All rights reserved.

This software is the confidential and proprietary 
information of Fujitsu Limited and Hiroshima University.
/*******************************************************/



#include <string>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <random>

#include "mp.h"
#include "host.h"
#include "device.cuh"
#include "constants.h"


#include <cuda.h>




int main(int argc, char* argv[])
{
    // usage
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] \
                  << " <host or device> <single or incremental> <run or test> <#inputs> <max n> <max x>" \
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // parameters for bulk evaluation of the Boys function
    const std::string HorD = argv[1];
    const std::string scenario = argv[2];
    const std::string mode = argv[3];
    const int num_inputs = std::pow(2, std::stoi(argv[4]));
    const Degree n_max = std::stoi(argv[5]); 
    const double x_max = std::stod(argv[6]);

    // int num_inputs = 1U << 22;
    // const Degree n_max = 24;
    // const double x_max = 40.0;

    // input parameter check
    if (num_inputs > 1U << 27 || n_max > 24 || x_max < 0.0 || x_max > 40.0) {
        std::cerr << "Error: inputs out of range" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    printf("\n");
    std::cout << HorD << " " << scenario << " evaluation" << std::endl;
    printf("---<INPUT PARAMETERS>---\n");
    printf("#inputs: %d\n", num_inputs);
    printf("n range: 0 <= n <= %d\n", n_max);
    printf("x range: 0.0 <= x <= %.1lf\n", x_max);

    Degree* n;
    double* x;
    cudaMallocHost(&n, sizeof(Degree) * num_inputs);
    cudaMallocHost(&x, sizeof(double) * num_inputs);
    TaylorTable LUT(n_max, LUT_K_MAX, LUT_XI_MAX, LUT_XI_INTERVAL);

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_int_distribution<Degree> dist_n(0, n_max);
    std::uniform_real_distribution<double> dist_x(0.0, x_max);

    int num_evals = 0;
    for (int i = 0; i < num_inputs; i++) {
        //n[i] = n_max;
        n[i] = dist_n(engine);
        x[i] = dist_x(engine);
        num_evals += n[i] + 1;    // for incremental evaluation scenario
    }

    double* boys;
    if (scenario == "single") {
        num_evals = num_inputs;
        cudaMallocHost(&boys, sizeof(double) * num_inputs);
    }
    else if (scenario == "incremental") {
        cudaMallocHost(&boys, sizeof(double) * (n_max + 1) * num_inputs);
    }
    else {
        std::cerr << "Error: scenario is \"single\" or \"incremental\"" \
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (HorD == "host") {
        hostBulkEvaluation(n, x, boys, scenario, LUT, num_inputs, num_evals);
    }
    else if (HorD == "device") {
        deviceBulkEvaluation(n, x, boys, scenario, LUT, num_inputs, num_evals);
    }
    else {
        std::cerr << "Error: \"host\" or \"device\"" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (mode == "test") {
        const double error_tol = 1.0e-14;
        testBoysEvaluation(n, x, boys, num_inputs, n_max, error_tol, scenario);
    }

    cudaFreeHost(n);
    cudaFreeHost(x);
    cudaFreeHost(boys);

    return 0;
}



