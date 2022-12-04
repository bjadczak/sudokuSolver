#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "static.cuh"
#include "libs.cuh"
#include "kernel.cuh"
#include "handler.cuh"

__host__ int solveSudoku(int *start_board);
