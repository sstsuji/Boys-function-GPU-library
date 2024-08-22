/*******************************************************
Copyright Fujitsu Limited and Hiroshima University 2023
All rights reserved.

This software is the confidential and proprietary 
information of Fujitsu Limited and Hiroshima University.
/*******************************************************/


#ifndef MP_H
#define MP_H

#include "gmp.h"
#include "constants.h"

struct TaylorTable
{
    const Degree n_max;
    const Degree k_max;
    const double xi_max;
    const double xi_interval;
    int num_xi;
    size_t table_size;
    double* boys_grid;

    TaylorTable(Degree n_max, Degree k_max, double xi_max, double xi_interval);
    ~TaylorTable();
};

void mpTaylorExpansion(Degree n, double x, Degree k_max, mpf_t* boys_true);

int compareTwoMpValues(mpf_t val_true, mpf_t val_double, mpf_t error_tol);

void testBoysEvaluation(Degree* ns, double* xs, double* boys, int num_inputs, 
                        Degree n_max, double error_tol_d, std::string scenario);


#endif // MP_H