/*******************************************************
Copyright Fujitsu Limited and Hiroshima University 2023
All rights reserved.

This software is the confidential and proprietary 
information of Fujitsu Limited and Hiroshima University.
/*******************************************************/


#ifndef HOST_H
#define HOST_H

#include "constants.h"

void hostBulkEvaluation(Degree* n, double* x, double* boys, 
                        std::string scenario, const TaylorTable& LUT, 
                        int num_inputs, int num_evals);

void hostSingleEvaluation(Degree* n, double* x, double* boys, 
                          const TaylorTable& LUT, int num_inputs);

void hostIncrementalEvaluation(Degree* n, double* x, double* boys, 
                               const TaylorTable& LUT, int num_inputs);


#endif // HOST_H