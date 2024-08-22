#********************************************************#
# Copyright Fujitsu Limited and Hiroshima University 2023
# All rights reserved.
#
# This software is the confidential and proprietary 
# information of Fujitsu Limited and Hiroshima University.
#********************************************************#


#!/bin/bash

binary=${1}

for i in {15..27}
do
    #OMP_NUM_THREADS=$(nproc) ../bin/${binary} host single run ${i} 24 40.0
    #OMP_NUM_THREADS=$(nproc) ../bin/${binary} host incremental run ${i} 24 40.0
    #../bin/${binary} device single run ${i} 24 40.0
    ../bin/${binary} device incremental run ${i} 24 40.0
done