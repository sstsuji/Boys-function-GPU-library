#********************************************************#
# Copyright Fujitsu Limited and Hiroshima University 2023
# All rights reserved.
#
# This software is the confidential and proprietary 
# information of Fujitsu Limited and Hiroshima University.
#********************************************************#


#!/bin/bash

cd ..

binary=${1}

xi_intervals=()
num_xis=()
k_maxes=(8, 7, 6, 5, 5, 4, 4, 3)
for i in {2..9}
do
    xi_interval=$(echo "scale=10; 2^(-${i})" | bc)
    num_xi=$(echo "scale=0; 32/${xi_interval}" | bc)
    xi_intervals+=($xi_interval)
    num_xis+=($num_xi)
done
#echo "xi intervals: ${xi_intervals[@]}"
#echo "#xis: ${num_xis[@]}"

for j in "${!k_maxes[@]}"
do
    make -s -B BIN=${binary} \
         XI_INTERVAL=${xi_intervals[${j}]} \
         NUM_XI=${num_xis[${j}]} \
         K_MAX=${k_maxes[${j}]}
    
    ./bin/${binary} device single test 15 24 7.5
    #./bin/${binary} device incremental run 22 24 7.5
done

cd -