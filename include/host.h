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