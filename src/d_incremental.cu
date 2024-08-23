/*******************************************************
Copyright Fujitsu Limited and Hiroshima University 2023
All rights reserved.
/*******************************************************/


#include <cmath>
#include <cuda.h>
#include "constants.h"


__device__
void iOriginTaylorExpansion(Degree n_prime, Degree n, double x, 
                            double* g_boys, int num_inputs, int iid)
{
    double boys;
    double numerator;
    double factorial;
    // const Degree k_max = 12;
    const Degree k_max = __double2uint_rd(A * x*x*x - B * x*x + C * x + D);

    for (Degree j = n_prime; j <= n; ++j) {
        numerator = 1.0;
        factorial = 1.0;
        boys = __drcp_rn(2 * j + 1);    // k = 0  
        for (Degree k = 1; k <= k_max; ++k) {
            numerator *= -x;
            factorial *= k;
            boys += numerator / (factorial * (2 * j + 2 * k + 1));
        }
        g_boys[num_inputs * j + iid] = boys;
    }
}


__device__
void iGriddedTaylorExpansion(Degree n, double x, double* g_boys, 
                             double* g_table, int num_inputs, int iid)
{
    double boys;
    double numerator;
    int factorial;
    const int x_idx = __double2uint_rd(x / LUT_XI_INTERVAL + 0.5);
    const double delta_x = x - (LUT_XI_INTERVAL * x_idx);
    
    for (Degree j = 0; j <= n; ++j) {
        numerator = 1.0;
        factorial = 1;
        boys = g_table[LUT_NUM_XI * j + x_idx];
        // boys = g_table[(25 + LUT_K_MAX) * x_idx + j];    // (x, n)

        for (Degree k = 1; k <= LUT_K_MAX; ++k) {
            numerator *= -(delta_x);
            factorial *= k;
            boys += (g_table[LUT_NUM_XI * (j + k) + x_idx] * numerator) / factorial;
            // boys += (g_table[(25 + LUT_K_MAX) * x_idx + (j + k)] * numerator) / factorial;
        }
        g_boys[num_inputs * j + iid] = boys;
    }
}


__device__
void iRecurrenceSemiInfinite(Degree n, double x, double* g_boys, 
                             Degree method, int num_inputs, int iid)
{
    double exp_neg_x = 0.0;
    const double reciprocal_double_x = __drcp_rn(2 * x);
    double boys = 0.5 * __dsqrt_rn(M_PI / x);    // j = 0

    // Recurrence relation method
    if (method >= 1) {
        exp_neg_x = exp(-x);
        boys *= erf(__dsqrt_rn(x));
    }

    g_boys[iid] = boys;
    for (Degree j = 1; j <= n; ++j) {
        boys = __fma_rn((2 * j - 1), boys, -exp_neg_x) * reciprocal_double_x;
        g_boys[num_inputs * j + iid] = boys;
    }
}


__global__
void incrementalBulkEvaluation(Sortkey* g_key, double* g_x, double* g_boys, 
                               double* g_table, int* g_cnt, int num_inputs)
{
    __shared__ int s_pid;
    const int tid = blockDim.x * threadIdx.y + threadIdx.x; 

    if (tid == 0) {
        s_pid = atomicAdd(g_cnt, 1);
    }
    __syncthreads();

    const int pid = s_pid;
    const int iid = blockDim.x * blockDim.y * pid + tid;

    const Sortkey key = g_key[iid];
    const Degree n = key & 0b00111111;
    const Degree method = (key & 0b11000000) >> 6;
    const double x = g_x[iid];

    //*
    if (x == 0.0) {
        for (Degree j = 0; j <= n; ++j) {
            g_boys[num_inputs * j + iid] = __drcp_rn(2 * j + 1);
        }
    }
    else if (method == 2) {
        iGriddedTaylorExpansion(n, x, g_boys, g_table, num_inputs, iid);
    }
    else {
        iRecurrenceSemiInfinite(n, x, g_boys, method, num_inputs, iid);
    }
    /**/

    //iGriddedTaylorExpansion(n, x, g_boys, g_table, num_inputs, iid);
}

