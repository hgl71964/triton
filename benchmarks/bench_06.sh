#! /bin/bash

# config
start=${1:-1}
seeds=${2:-100}
workloads=(
        16384
        8192
        4096
        2048
        1024
        512
        # 32768
     )

for seed in $( seq $start $seeds ); do
        for workload in "${workloads[@]}"; do
                echo
                echo "workload ${workload}; seed ${seed}: "
                echo
                python benchmarks/06-fused-attention.py \
                        --seed $seed \
                        --ctx $workload
        done
done
