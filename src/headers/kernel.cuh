#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "static.cuh"

__device__ void calculatePossibilites(const int *currentBoard, int *emptyCells, possibilitie *poss, int *possCount);
__global__ void runSolver(const int *currentBoard, possibleBoard *possBoard);