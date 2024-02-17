#! /bin/bash

# config
test=500
start=${1:-1}
factor=4
workloads=(
        64
        256
        512
        1024
        2048
        4096

        8192
        16384
     )

for workload in "${workloads[@]}"; do
        echo
        echo "workload ${workload}; seed ${seed}: "
        echo
        python benchmarks/03-matrix-multiplication.py \
                --n_tests $test \
                --factor $factor \
                --wl $workload \
                --load 1 \
                --bench 1
        sleep 3
done
