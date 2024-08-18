#include <cstdio>
#include <cstdint>
//#include <cmath>
#include <iostream>

#include "gmp.h"
#include <cuda.h>

#include "mp.h"
#include "constants.h"


void mpTaylorExpansion(Degree n, double x, Degree k_max, mpf_t* boys_true)
{
    mpf_t neg_x;
    mpf_init_set_d(neg_x, -x);

    mpf_t numerator;
    mpf_t denominator;
    mpf_t factorial;
    mpf_t large_odd;

    mpf_init_set_d(numerator, 1.0);
    mpf_init(denominator);
    mpf_init_set_d(factorial, 1.0);
    mpf_init(large_odd);

    for (Degree j = 0; j <= n; j++) {
        mpf_set_ui(large_odd, (2 * j + 1));
        mpf_ui_div(boys_true[j], 1, large_odd);
    }

    for (Degree k = 1; k <= k_max; k++) {
        mpf_mul(numerator, numerator, neg_x);
        mpf_mul_ui(factorial, factorial, k);
        for (Degree j = 0; j <= n; j++) {
            mpf_mul_ui(denominator, factorial, (2 * j + 2 * k + 1));
            mpf_div(denominator, numerator, denominator);
            mpf_add(boys_true[j], boys_true[j],  denominator);
        }
    }

    mpf_clear(neg_x);
    mpf_clear(numerator);
    mpf_clear(denominator);
    mpf_clear(factorial);
    mpf_clear(large_odd);
}


TaylorTable::TaylorTable(Degree n_max, Degree k_max, 
                         double xi_max, double xi_interval)
    : n_max(n_max), 
      k_max(k_max), 
      xi_max(xi_max), 
      xi_interval(xi_interval)
{
    mpf_set_default_prec(128);

    num_xi = static_cast<int>(xi_max / xi_interval);
    const Degree num_columns = n_max + k_max;
    table_size = sizeof(double) * num_xi * (num_columns + 1);

    cudaMallocHost(&boys_grid, table_size);
    cudaMemset(boys_grid, 0, table_size);

    mpf_t boys_mp[num_columns + 1];
    for (Degree j = 0; j <= num_columns; j++) {
        mpf_init(boys_mp[j]);
    }
        
    double xi;
    for (int x_idx = 0; x_idx < num_xi; x_idx++) {
        xi = xi_interval * x_idx;
        mpTaylorExpansion(num_columns, xi, 200, boys_mp);
        for (Degree j = 0; j <= num_columns; j++) {
            boys_grid[num_xi * j + x_idx] = mpf_get_d(boys_mp[j]);    // (n, x)
            // boys_grid[(num_columns + 1) * x_idx + j] = mpf_get_d(boys_mp[j]);    // (x, n)
        }
    }

    for (Degree j = 0; j <= num_columns; j++) {
        mpf_clear(boys_mp[j]);
    }

    printf("---<LUT PARAMETERS>---\n");
    printf("x range: 0.0 <= x_i < %.1lf\n", xi_max);
    printf("x interval: d = %.5lf\n", xi_interval);
    printf("terms: k_max = %d\n", k_max);
    printf("table size: %.2lf [KB]\n", num_xi * (num_columns + 1) * 8 / 1e3);
}

TaylorTable::~TaylorTable()
{
    cudaFreeHost(boys_grid);
}


int compareTwoMpValues(mpf_t val_true, mpf_t val_double, mpf_t error_tol)
{
    int ppp = 0;
    mpf_t abs_error;
    mpf_init(abs_error);
    mpf_sub(abs_error, val_true, val_double);
    mpf_abs(abs_error, abs_error);

    ppp = mpf_cmp(error_tol, abs_error);

    mpf_clear(abs_error);

    return ppp;
}


void testBoysEvaluation(Degree* ns, double* xs, double* boys, int num_inputs, 
                        Degree n_max, double error_tol_d, std::string scenario)
{
    mpf_set_default_prec(128);
    printf("---<NUMERICAL TEST>---\n");

    mpf_t boys_mp;
    mpf_t error_tol_mp;
    mpf_init(boys_mp);
    mpf_init_set_d(error_tol_mp, error_tol_d);

    mpf_t boys_true[n_max + 1];
    for (Degree j = 0; j <= n_max; ++j) {
        mpf_init(boys_true[j]);
    }

    int ppp = -1;
    int num_checked = 0;
    float progress = 0.0;
    const int log_interval = num_inputs / 8;

    Degree n;
    double x;
    for (int i = 0; i < num_inputs; ++i) {
        n = ns[i];
        x = xs[i];
        mpTaylorExpansion(n, x, 200, boys_true);

        if (scenario == "single") {
            mpf_set_d(boys_mp, boys[i]);
            ppp = compareTwoMpValues(boys_true[n], boys_mp, error_tol_mp);
            if (ppp < 0) {
                std::cerr << "Error: numerical error over " << error_tol_d \
                          << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        else if (scenario == "incremental") {
            for (Degree j = 0; j <= n; ++j) {
                mpf_set_d(boys_mp, boys[num_inputs * j + i]);
                ppp = compareTwoMpValues(boys_true[j], boys_mp, error_tol_mp);
                if (ppp < 0) {
                    std::cerr << "Error: numerical error over " << error_tol_d \
                              << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
        }
        num_checked++;
        
        if ((i % log_interval) == 0) {
            progress = (static_cast<float>(i) / num_inputs) * 100;
            printf("%.1f%% completed.\n", progress);
            fflush(stdout);
        }
    }

    if (num_checked == num_inputs) {
        printf("passed.\n");
    }

    mpf_clear(boys_mp);
    mpf_clear(error_tol_mp);
    for (Degree j = 0; j <= n_max; ++j) {
        mpf_clear(boys_true[j]);
    }
}



