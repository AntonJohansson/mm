#!/bin/sh

clang matmul.c -O3 -g -march=native -fopenmp -lm -lcblas
#gcc -g matmul.c -O3 -g -fsanitize=address -march=native -fopenmp -lm
