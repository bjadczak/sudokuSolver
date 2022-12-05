#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "static.cuh"
#include "libs.cuh"
#include "kernel.cuh"
#include "handler.cuh"
#include "sudokuSolverException.cuh"

const auto cmpQueue = [](possibleBoard left, possibleBoard right)
{ return (left.status) > (right.status); };

__host__ int solveSudoku(int *start_board);
__host__ void prepareDevice(cudaError_t &cudaStatus, void **poss_d, void **sudokuBoard) throw();
__host__ void addNewBoardsToDevice(int &indx, std::priority_queue<possibleBoard, std::vector<possibleBoard>, decltype(cmpQueue)> &Q, int *sudokuBoard, cudaError_t &cudaStatus) throw();
__host__ int *addNewBoardsToQueue(int &indx, possibleBoard *poss_h, std::priority_queue<possibleBoard, std::vector<possibleBoard>, decltype(cmpQueue)> &Q);
__host__ void addToQueue(std::priority_queue<possibleBoard, std::vector<possibleBoard>, decltype(cmpQueue)> &Q, int *board, int status);
__host__ void fetchResoults(cudaError_t &cudaStatus, possibleBoard *poss_h, possibleBoard *poss_d) throw();