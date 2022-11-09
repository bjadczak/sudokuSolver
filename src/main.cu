#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define N 3
#define BOARD_SIZE (N * N)
#define CELL_COUNT (BOARD_SIZE * BOARD_SIZE)

// #define DEBUG_MODE

__device__ void printBoard(int *sudokuBoard, int *targetCell, int id)
{
    printf("TARGET CELL - %d (id: %d)\n", targetCell[id], id);
}

__global__ void fillEmpty(int *sudokuBoard, int *targetCell)
{
#ifdef DEBUG_MODE
    printBoard(sudokuBoard, targetCell, threadIdx.x);
#endif

    // Tables we use to count appearence
    int appeardInRow[BOARD_SIZE] = {0};
    int appeardInColumn[BOARD_SIZE] = {0};
    int appeardInBlock[BOARD_SIZE] = {0};

    int target = targetCell[threadIdx.x];

    for (int i = 0; i < BOARD_SIZE; i++)
    {
        int tmp = sudokuBoard[target / BOARD_SIZE + i];
        if (tmp > 0)
        {
            appeardInRow[tmp - 1]++;
        }
        // printf("ROW - cell: %d is %d\n", target / BOARD_SIZE + i, tmp);
    }

    for (int i = 0; i < BOARD_SIZE; i++)
    {

        int tmp = sudokuBoard[target % BOARD_SIZE + i * BOARD_SIZE];
        if (tmp > 0)
        {
            appeardInColumn[tmp - 1]++;
        }
        // printf("COLUMN - cell: %d is %d\n", target % BOARD_SIZE + i * BOARD_SIZE, tmp);
    }

    int firstCellOfBlock = ((target / BOARD_SIZE) / N) * BOARD_SIZE * N + ((target % BOARD_SIZE) / N) * N;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int tmp = sudokuBoard[firstCellOfBlock + i * BOARD_SIZE + j];
            if (tmp > 0)
            {
                appeardInBlock[tmp - 1]++;
            }
            // printf("BLOCK - cell: %d is %d\n", firstCellOfBlock + i * BOARD_SIZE + j, tmp);
        }
    }

#ifdef DEBUG_MODE
    for (int i = 0; i < BOARD_SIZE; i++)
    {
        printf("%d", appeardInRow[i]);
    }
    printf(" - Appeared in row; Target - %d\n", target);

    for (int i = 0; i < BOARD_SIZE; i++)
    {
        printf("%d", appeardInColumn[i]);
    }
    printf(" - Appeared in column; Target - %d\n", target);

    for (int i = 0; i < BOARD_SIZE; i++)
    {
        printf("%d", appeardInBlock[i]);
    }
    printf(" - Appeared in block; Target - %d\n", target);
#endif
}

int main()
{
    cudaError_t cudaStatus;

    const int start_board[CELL_COUNT] =
        {
            3,
            0,
            0,
            8,
            0,
            1,
            0,
            0,
            2,
            2,
            0,
            1,
            0,
            3,
            0,
            6,
            0,
            4,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            8,
            0,
            9,
            0,
            0,
            0,
            1,
            0,
            6,
            0,
            6,
            0,
            0,
            0,
            0,
            0,
            5,
            0,
            7,
            0,
            2,
            0,
            0,
            0,
            4,
            0,
            9,
            0,
            0,
            0,
            5,
            0,
            9,
            0,
            0,
            0,
            9,
            0,
            4,
            0,
            8,
            0,
            7,
            0,
            5,
            6,
            0,
            0,
            0,
            0,
            7,
            0,
            0,
            3,
        };
    int empty_cells[CELL_COUNT] = {-1};

    int indx = 0;
    for (int i = 0; i < CELL_COUNT; i++)
        if (start_board[i] == 0)
        {
            empty_cells[indx] = i;
            indx++;
        }

    int *sudokuBoard = 0;
    int *targetCell = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void **)&sudokuBoard, CELL_COUNT * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void **)&targetCell, CELL_COUNT * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(sudokuBoard, start_board, CELL_COUNT * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(targetCell, empty_cells, CELL_COUNT * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    fillEmpty<<<1, indx>>>(sudokuBoard, targetCell);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

Error:
    cudaFree(sudokuBoard);

    return cudaStatus;
}