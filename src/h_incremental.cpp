#include <cmath>
#include <omp.h>

#include "mp.h"
#include "constants.h"


Degree selectMethodIncremental(Degree n, double x)
{
    Degree method;
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
    return method;
}


void hiOriginTaylorExpansion(Degree n_prime, Degree n, double x, 
                             double* boys, int num_inputs, int iid)
{
    double tmp;
    double numerator = 1.0;
    double factorial = 1.0;
    const Degree k_max = static_cast<int>(A * x*x*x - B * x*x + C * x + D);

    for (Degree j = n_prime; j <= n; ++j) {
        numerator = 1.0;
        factorial = 1.0;
        tmp = 1 / static_cast<double>(2 * j + 1);    // k = 0  
        for (Degree k = 1; k <= k_max; ++k) {
            numerator *= -x;
            factorial *= k;
            tmp += numerator / (factorial * (2 * j + 2 * k + 1));
        }
        boys[num_inputs * j + iid] = tmp;
    }
}

void hiGriddedTaylorExpansion(Degree n, double x, double* boys, 
                              double* table, int num_inputs, int iid)
{
    double tmp;
    double numerator;
    int factorial;
    const int x_idx = std::floor(x / LUT_XI_INTERVAL + 0.5);
    const double delta_x = x - (LUT_XI_INTERVAL * x_idx);
    
    for (Degree j = 0; j <= n; ++j) {
        numerator = 1.0;
        factorial = 1;
        tmp = table[LUT_NUM_XI * j + x_idx];

        for (Degree k = 1; k <= LUT_K_MAX; ++k) {
            numerator *= -(delta_x);
            factorial *= k;
            tmp += (table[LUT_NUM_XI * (j + k) + x_idx] * numerator) / factorial;
        }
        boys[num_inputs * j + iid] = tmp;
    }
}


void hiRecurrenceSemiInfinite(Degree n, double x, double* boys, 
                              Degree method, int num_inputs, int iid)
{
    double exp_neg_x = 0.0; 
    const double reciprocal_double_x = 1 / (2 * x);
    double tmp = 0.5 * std::sqrt(M_PI / x);    // j = 0;

    // Recurrence relation method
    if (method >= 1) {
        exp_neg_x = std::exp(-x);
        tmp *= std::erf(std::sqrt(x));
    }

    boys[iid] = tmp;
    for (Degree j = 1; j <= n; ++j) {
        tmp = ((2 * j - 1) * tmp - exp_neg_x) * reciprocal_double_x;
        boys[num_inputs * j + iid] = tmp;
    }
}


void hostIncrementalEvaluation(Degree* n, double* x, double* boys, 
                               const TaylorTable& LUT, int num_inputs)
{
    #pragma omp parallel for
    for (int i = 0; i < num_inputs; ++i) {
        Degree method = selectMethodIncremental(n[i], x[i]);

        if (x[i] == 0.0) {
            for (Degree j = 0; j <= n[i]; ++j) {
                boys[num_inputs * j + i] = 1 / static_cast<double>(2 * j + 1);
            }
        }
        else if (method == 2) {
            hiGriddedTaylorExpansion(n[i], x[i], boys, LUT.boys_grid, num_inputs, i);
        }
        else {
            hiRecurrenceSemiInfinite(n[i], x[i], boys, method, num_inputs, i);
        }
    }
}



