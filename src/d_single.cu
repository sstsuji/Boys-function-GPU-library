#include <cmath>
#include <cuda.h>
#include "constants.h"


__device__
double sOriginTaylorExpansion(Degree n, double x)
{
    double numerator = 1.0;
    double factorial = 1.0;
    double boys = __drcp_rn(2 * n + 1);    // k = 0;
    // const Degree k_max = 12;
    const Degree k_max = __double2uint_rd(A * x*x*x - B * x*x + C * x + D);
    
    for (Degree k = 1; k <= k_max; ++k) {
        numerator *= -x;
        factorial *= k;
        boys += numerator / (factorial * (2 * n + 2 * k + 1));
    }
    return boys;
}


__device__
double sGriddedTaylorExpansion(Degree n, double x, double* g_table)
{
    double numerator = 1.0;
    int factorial = 1;
    const int x_idx = __double2uint_rd(x / LUT_XI_INTERVAL + 0.5);
    const double delta_x = x - (LUT_XI_INTERVAL * x_idx);
    double boys = g_table[LUT_NUM_XI * n + x_idx];    // (n, x)
    //double boys = g_table[(25 + LUT_K_MAX) * x_idx + n];    // (x, n)

    for (Degree k = 1; k <= LUT_K_MAX; ++k) {
        numerator *= -(delta_x);
        factorial *= k;
        boys += (g_table[LUT_NUM_XI * (n + k) + x_idx] * numerator) / factorial;
        //boys += (g_table[(25 + LUT_K_MAX) * x_idx + (n + k)] * numerator) / factorial;    // (x, n)
    }
    return boys;
}


__device__
double sRecurrenceRelation(Degree n, double x)
{
    double exp_neg_x = 0.0; 
    const double reciprocal_double_x = __drcp_rn(2 * x);
    double boys = 0.5 * __dsqrt_rn(M_PI / x);    // j = 0;

    exp_neg_x = exp(-x);
    boys *= erf(__dsqrt_rn(x));
    for (Degree j = 1; j <= n; ++j) {
        boys = __fma_rn((2 * j - 1), boys, -exp_neg_x) * reciprocal_double_x;
    }
    return boys;
}


__device__
double sSemiInfiniteInterval(Degree n, double x)
{
    const double reciprocal_double_x = __drcp_rn(2 * x);
    double boys = 0.5 * __dsqrt_rn(M_PI / x);    // j = 0;

    for (Degree j = 1; j <= n; ++j) {
        boys = (2 * j - 1) * boys * reciprocal_double_x;
    }
    return boys;
}



__global__
void singleBulkEvaluation(Sortkey* g_key, double* g_x, 
                          double* g_boys, double* g_table, int* g_cnt)
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
        g_boys[iid] = __drcp_rn(2 * n + 1);
    }
    else if (method == 1) {
        g_boys[iid] = sGriddedTaylorExpansion(n, x, g_table);
    }
    else {
        g_boys[iid] = sSemiInfiniteInterval(n, x);
    }
    /**/

    //g_boys[iid] = sGriddedTaylorExpansion(n, x, g_table);
}



