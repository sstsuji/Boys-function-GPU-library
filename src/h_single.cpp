#include <cmath>
#include <omp.h>

#include "mp.h"
#include "constants.h"



Degree selectMethodSingle(Degree n, double x)
{
    return x < (A_RS * n + B_RS);
}


double hsOriginTaylorExpansion(Degree n, double x)
{
    double numerator = 1.0;
    double factorial = 1.0;
    double boys = 1 / static_cast<double>(2 * n + 1);
    const Degree k_max = static_cast<int>(A * x*x*x - B * x*x + C * x + D);
    
    for (Degree k = 1; k <= k_max; ++k) {
        numerator *= -x;
        factorial *= k;
        boys += numerator / (factorial * (2 * n + 2 * k + 1));
    }
    return boys;
}


double hsGriddedTaylorExpansion(Degree n, double x, double* table)
{
    double numerator = 1.0;
    int factorial = 1;
    const int x_idx = std::floor(x / LUT_XI_INTERVAL + 0.5);
    const double delta_x = x - (LUT_XI_INTERVAL * x_idx);
    double boys = table[LUT_NUM_XI * n + x_idx];

    for (Degree k = 1; k <= LUT_K_MAX; ++k) {
        numerator *= -(delta_x);
        factorial *= k;
        boys += (table[LUT_NUM_XI * (n + k) + x_idx] * numerator) / factorial;
    }
    return boys;
}


double hsRecurrenceRelation(Degree n, double x)
{
    double exp_neg_x = 0.0; 
    const double reciprocal_double_x = 1 / (2 * x);
    double boys = 0.5 * std::sqrt(M_PI / x);    // j = 0;
    exp_neg_x = std::exp(-x);
    boys *= std::erf(std::sqrt(x));

    for (Degree j = 1; j <= n; ++j) {
        boys = ((2 * j - 1) * boys - exp_neg_x) * reciprocal_double_x;
    }
    return boys;
}


double hsSemiInfiniteInterval(Degree n, double x)
{
    double exp_neg_x = 0.0; 
    const double reciprocal_double_x = 1 / (2 * x);
    double boys = 0.5 * std::sqrt(M_PI / x);    // j = 0;

    for (Degree j = 1; j <= n; ++j) {
        boys = ((2 * j - 1) * boys - exp_neg_x) * reciprocal_double_x;
    }
    return boys;
}


void hostSingleEvaluation(Degree* n, double* x, double* boys, 
                          const TaylorTable& LUT, int num_inputs) 
{
    #pragma omp parallel for
    for (int i = 0; i < num_inputs; ++i) {
        Degree method = selectMethodSingle(n[i], x[i]);

        if (x[i] == 0.0) {
            boys[i] = 1 / static_cast<double>(2 * n[i] + 1);
        }
        else if (method == 1) {
            boys[i] = hsGriddedTaylorExpansion(n[i], x[i], LUT.boys_grid);
        }
        else {
            boys[i] = hsSemiInfiniteInterval(n[i], x[i]);
        }
    }
}






