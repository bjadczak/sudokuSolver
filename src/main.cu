#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define N 3
#define BOARD_SIZE (N * N)
#define CELL_COUNT (BOARD_SIZE * BOARD_SIZE)

#define DEBUG_MODE

struct appeared
{
    int cell;

    int appeardInRow[BOARD_SIZE] = {0};
    int appeardInColumn[BOARD_SIZE] = {0};
    int appeardInBlock[BOARD_SIZE] = {0};

} typedef appeared;

__host__ void printBoard(appeared *app)
{
    for (int i = 0; i < CELL_COUNT; i++)
    {

        printf("[%d]TARGET CELL - %d\n", i, app[i].cell);

        for (int j = 0; j < BOARD_SIZE; j++)
        {
            printf("%d", app[i].appeardInRow[j]);
        }
        printf(" - Appeared in row\n");

        for (int j = 0; j < BOARD_SIZE; j++)
        {
            printf("%d", app[i].appeardInColumn[j]);
        }
        printf(" - Appeared in column\n");

        for (int j = 0; j < BOARD_SIZE; j++)
        {
            printf("%d", app[i].appeardInBlock[j]);
        }
        printf(" - Appeared in block\n");
    }
}

__global__ void fillEmpty(const int *sudokuBoard, const int *targetCell, appeared *app)
{
#ifdef DEBUG_MODE

#endif

    sudokuBoard += threadIdx.x * CELL_COUNT;
    // app += threadIdx.x * sizeof(appeared);
    app[threadIdx.x].cell = targetCell[threadIdx.x];

    // Calculate notes -- if it turns out there is only one possiblity - insert it

    for (int i = 0; i < BOARD_SIZE; i++)
    {
        int tmp = sudokuBoard[app[threadIdx.x].cell / BOARD_SIZE + i];
        if (tmp > 0)
        {
            app[threadIdx.x].appeardInRow[tmp - 1]++;
        }
        // printf("ROW - cell: %d is %d\n", app.cell / BOARD_SIZE + i, tmp);
    }

    for (int i = 0; i < BOARD_SIZE; i++)
    {

        int tmp = sudokuBoard[app[threadIdx.x].cell % BOARD_SIZE + i * BOARD_SIZE];
        if (tmp > 0)
        {
            app[threadIdx.x].appeardInColumn[tmp - 1]++;
        }
        // printf("COLUMN - cell: %d is %d\n", app->cell % BOARD_SIZE + i * BOARD_SIZE, tmp);
    }

    int firstCellOfBlock = ((app[threadIdx.x].cell / BOARD_SIZE) / N) * BOARD_SIZE * N + ((app[threadIdx.x].cell % BOARD_SIZE) / N) * N;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int tmp = sudokuBoard[firstCellOfBlock + i * BOARD_SIZE + j];
            if (tmp > 0)
            {
                app[threadIdx.x].appeardInBlock[tmp - 1]++;
            }
            // printf("BLOCK - cell: %d is %d\n", firstCellOfBlock + i * BOARD_SIZE + j, tmp);
        }
    }

    // Check what notes we have

    int number = -1;
    int numOfZerosRow = 0;
    int numOfZerosColumn = 0;
    int numOfZerosBlock = 0;

    for (int i = 0; i < BOARD_SIZE; i++)
    {

        // if ((appeardInRow[i] == 0 && appeardInColumn[i] == 0 && appeardInBlock[i] == 0) && number == -1)
        //     number = i + 1;
        // else if ((appeardInRow[i] == 0 && appeardInColumn[i] == 0 && appeardInBlock[i] == 0) && number != -1)
        //{
        //     number = -2;
        // }

        if (app[threadIdx.x].appeardInRow[i] == 0)
            numOfZerosRow++;
        if (app[threadIdx.x].appeardInColumn[i] == 0)
            numOfZerosColumn++;
        if (app[threadIdx.x].appeardInBlock[i] == 0)
            numOfZerosBlock++;
    }
#ifdef DEBUG_MODE
    printf("TARGET CELL - %d (id: %d); number of zeros row: %d; column: %d; block: %d\n", app->cell + 1, threadIdx.x, numOfZerosRow, numOfZerosColumn, numOfZerosBlock);
    // printf("TARGET CELL - %d (id: %d); number: %d\n", target + 1, threadIdx.x, number);
#endif
    __syncthreads();
}

int main()
{
    cudaError_t cudaStatus;

    // const int start_board[CELL_COUNT] =
    //     {
    //         3, 0, 0, 8, 0, 1, 0, 0, 2,
    //         2, 0, 1, 0, 3, 0, 6, 0, 4,
    //         0, 0, 0, 0, 1, 0, 0, 0, 0,
    //         8, 0, 9, 0, 0, 0, 1, 0, 6,
    //         0, 6, 0, 0, 0, 0, 0, 5, 0,
    //         7, 0, 2, 0, 0, 0, 4, 0, 9,
    //         0, 0, 0, 5, 0, 9, 0, 0, 0,
    //         9, 0, 4, 0, 8, 0, 7, 0, 5,
    //         6, 0, 0, 0, 0, 7, 0, 0, 3,
    //     };

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
    appeared *app = 0;

    appeared calculated[CELL_COUNT];

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void **)&sudokuBoard, CELL_COUNT * CELL_COUNT * sizeof(int));
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

    cudaStatus = cudaMalloc((void **)&app, CELL_COUNT * sizeof(appeared));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    for (int i = 0; i < CELL_COUNT; i++)
    {
        cudaStatus = cudaMemcpy((sudokuBoard + i * CELL_COUNT), start_board, CELL_COUNT * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
    }

    cudaStatus = cudaMemcpy(targetCell, empty_cells, CELL_COUNT * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    fillEmpty<<<1, indx>>>(sudokuBoard, targetCell, app);

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
    cudaStatus = cudaMemcpy(calculated, app, CELL_COUNT * sizeof(appeared), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    printBoard(calculated);

Error:
    cudaFree(sudokuBoard);
    cudaFree(targetCell);
    cudaFree(app);

    return cudaStatus;
}