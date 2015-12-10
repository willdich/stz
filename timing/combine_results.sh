#!/bin/bash

for n in 16 32 64 128 256
do
    for p in 1 2 4 8 16 32 64
    do
        cat time-${p}_${n}.out >> all-times-$n
    done
done
