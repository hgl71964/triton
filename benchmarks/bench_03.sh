#! /bin/bash

# config
start=${1:-1}
seeds=${2:-100}
workloads=(
        64
        256
        512
        1024
        2048
        4096
     )

for seed in $( seq $start $seeds ); do
        for workload in "${workloads[@]}"; do
                echo
                echo "workload ${workload}; seed ${seed}: "
                echo
                python benchmarks/03-matrix-multiplication.py \
                        --seed $seed \
                        --wl $workload
        done
done
