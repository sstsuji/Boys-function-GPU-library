/*******************************************************
Copyright Fujitsu Limited and Hiroshima University 2023
All rights reserved.

This software is the confidential and proprietary 
information of Fujitsu Limited and Hiroshima University.
/*******************************************************/


#ifndef DEVICE_CUH
#define DEVICE_CUH


#include <cuda.h>
#include "constants.h"

void deviceBulkEvaluation(Degree* h_n, double* h_x, double* h_boys, 
                          std::string scenario, const TaylorTable& h_LUT, 
                          int num_inputs, int num_evals);

__global__
void generateKeySingle(Degree* g_n, double* g_x, Sortkey* g_key);
__global__
void generateKeyIncremental(Degree* g_n, double* g_x, Sortkey* g_key);

__global__
void singleBulkEvaluation(Sortkey* g_key, double* g_x, 
                          double* g_boys, double* g_boys_grid, int* g_cnt);
__global__
void incrementalBulkEvaluation(Sortkey* g_key, double* g_x, double* g_boys, 
                               double* g_table, int* g_cnt, int num_inputs);


#endif // DEVICE_CUH