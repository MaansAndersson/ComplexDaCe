#!/bin/bash
export OMP_NUM_THREADS=1




for i in $(seq 1 1 21)
do
N=$((2 ** $i))
./a.out ${N} > log${N}
done
