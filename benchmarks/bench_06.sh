#! /bin/bash

# config
test=500
workloads=(
        16384
        8192
        4096
        2048
        1024
        512
        # 32768
     )

Zs=(
        1
        16
        # 128
     )

Hs=(
        16
        64
        4
     )

HEADs=(
        32
        64
        128
     )

for seed in $( seq 1 10 ); do
        for Z in "${Zs[@]}"; do
                for H in "${Hs[@]}"; do
                        for HEAD in "${HEADs[@]}"; do
                                for workload in "${workloads[@]}"; do
                                        echo
                                        echo "workload ${Z}_${H}_${workload}_${HEAD}; "
                                        echo
                                        python benchmarks/06-fused-attention.py \
                                                --seed $seed \
                                                --n_tests $test \
                                                --Z $Z \
                                                --H $H \
                                                --D_HEAD $HEAD \
                                                --wl $workload \
                                                --load 1 \
                                                --bench 1
                                        sleep 3
                                done
                        done
                done
        done
done