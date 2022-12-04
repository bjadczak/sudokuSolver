#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "static.cuh"
#include "libs.cuh"

__host__ void printBoard(int *sudokuBoard);
__host__ bool isBoardValid(int *sudokuBoard);
__host__ void loadBoard(int *board, std::ifstream &inFile);