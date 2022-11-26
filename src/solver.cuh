#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>

#include <stack>
#include <queue>
#include <chrono>

#define N 3
#define BOARD_SIZE (N * N)
#define CELL_COUNT (BOARD_SIZE * BOARD_SIZE)
#define NUM_OF_THREADS 1024
#define NUM_OF_BLOCKS 10
#define NUM_OF_KERNELS (NUM_OF_THREADS * NUM_OF_BLOCKS)

struct possibilitie
{
    int poss[BOARD_SIZE] = {0};
    int cell = -1;

} typedef possibilitie;
struct possibleBoard
{
    int board[CELL_COUNT] = {0};
    int status = 0;

} typedef possibleBoard;
struct board
{
    int board[CELL_COUNT] = {0};

} typedef board;

__host__ void printBoard(int *sudokuBoard);
__host__ __device__ bool isBoardValid(int *sudokuBoard);
__device__ void calculatePossibilites(const int *currentBoard, int *emptyCells, possibilitie *poss, int *possCount);
__global__ void runSolver(const int *currentBoard, possibleBoard *possBoard);
__host__ int solveSudoku(int *start_board);
__host__ void loadBoard(int *board, std::ifstream &inFile);