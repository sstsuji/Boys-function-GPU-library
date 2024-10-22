#********************************************************#
# Copyright Fujitsu Limited and Hiroshima University 2023
# All rights reserved.
#********************************************************#


#!/bin/bash

cd ..

binary=${1}

xi_intervals=(0.25, 0.125, 0.0625, 0.03125, 0.015625, \
             0.0078125, 0.00390625, 0.001953125)
num_xis=(128, 256, 512, 1024, 2048, 4096, 8192, 16384)
k_maxes=(8, 7, 6, 5, 5, 4, 4, 3)

#xi_intervals=()
#num_xis=()
#for i in {2..9}
#do
#    xi_interval=$(echo "scale=10; 2^(-${i})" | bc)
#    num_xi=$(echo "scale=0; 32/${xi_interval}" | bc)
#    xi_intervals+=($xi_interval)
#    num_xis+=($num_xi)
#done
#echo "xi intervals: ${xi_intervals[@]}"
#echo "#xis: ${num_xis[@]}"

for j in "${!k_maxes[@]}"
do
    #echo "in: ${xi_intervals[${j}]}"
    #echo "num: ${num_xis[${j}]}"
    #echo "j: ${k_maxes[${j}]}"
    make -s -B BIN=${binary}_${j} \
         XI_INTERVAL=${xi_intervals[${j}]} \
         NUM_XI=${num_xis[${j}]} \
         K_MAX=${k_maxes[${j}]}
    
    #./bin/${binary}_${j} device single test 15 24 7.5
    ./bin/${binary}_${j} device incremental run 22 24 7.5
done

cd -