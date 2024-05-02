#!/bin/sh

clang c/test.c -O3 -g -march=native -fopenmp -lm -lcblas -o mm-clang-c
gcc   c/test.c -O3 -g -march=native -fopenmp -lm -lcblas -o mm-gcc-c

clang++ cpp/test.cpp -std=c++23 -O3 -g -march=native -fopenmp -lm -lcblas -o mm-clang-cpp
g++     cpp/test.cpp -std=c++23 -O3 -g -march=native -fopenmp -lm -lcblas -o mm-gcc-cpp

#zig build-exe -OReleaseFast zig/test.zig --name mm-zig

#clang jl/rdtscp.c -O3 -g -march=native -shared -o rdtscp.so
#julia jl/test.jl --check-bounds=no
