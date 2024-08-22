/*******************************************************
Copyright Fujitsu Limited and Hiroshima University 2023
All rights reserved.

This software is the confidential and proprietary 
information of Fujitsu Limited and Hiroshima University.
/*******************************************************/


#include <cuda.h>
#include "constants.h"


__global__
void generateKeySingle(Degree *g_n, double* g_x, Sortkey* g_key)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const Degree n = g_n[tid];
    const double x = g_x[tid];

    const Sortkey method = x < (A_RS * n + B_RS);
    g_key[tid] = (method << 6) | n;
}


__global__
void generateKeyIncremental(Degree *g_n, double* g_x, Sortkey* g_key)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const Degree n = g_n[tid];
    const double x = g_x[tid];

    Sortkey method;
    if (x < (A_TR * n + B_TR)) {
        method = 2;    // Gridded Taylor expansion method
    }
    else {
        if (x < (A_RS * n + B_RS)) {
            method = 1;    // Recurrence relation method
        }
        else {
            method = 0;    // Semi-infinite interval method
        }
    }
    g_key[tid] = (method << 6) | n;
}



