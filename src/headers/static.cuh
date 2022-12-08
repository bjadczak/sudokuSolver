#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 3
#define BOARD_SIZE (N * N)
#define CELL_COUNT (BOARD_SIZE * BOARD_SIZE)
#define NUM_OF_THREADS 1024
#define NUM_OF_BLOCKS 10
#define NUM_OF_KERNELS (NUM_OF_THREADS * NUM_OF_BLOCKS)

// #define DEBUG_MODE
// #define PRINT_BOARDS

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