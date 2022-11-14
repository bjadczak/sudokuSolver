#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <queue>

#define N 3
#define BOARD_SIZE (N * N)
#define CELL_COUNT (BOARD_SIZE * BOARD_SIZE)

#define DEBUG_MODE

struct appeared
{
    int cell = -1;

    int appeardInRow[BOARD_SIZE] = {0};
    int appeardInColumn[BOARD_SIZE] = {0};
    int appeardInBlock[BOARD_SIZE] = {0};

} typedef appeared;

__global__ void fillEmpty(const int *sudokuBoard, const int *targetCell, appeared *app);

__host__ void printBoardInfo(appeared *app)
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
__host__ void printBoard(int *sudokuBoard)
{
    for (int i = 0; i < BOARD_SIZE; i++)
    {
        for (int j = 0; j < BOARD_SIZE; j++)
        {
            printf("%d ", sudokuBoard[i * BOARD_SIZE + j]);
        }
        printf("\n");
    }
}
__host__ bool isBoardValid(int *sudokuBoard)
{
    for (int i = 0; i < BOARD_SIZE; i++)
        {
            int all[BOARD_SIZE] = {0};
            for (int j = 0; j < BOARD_SIZE; j++)
            {
                if (sudokuBoard[i*BOARD_SIZE + j] > 0)
                    all[sudokuBoard[i*BOARD_SIZE + j] - 1]++;
            }
            for (int j = 0; j < BOARD_SIZE; j++)
                if (all[j] > 1)
                    return false;
        }
        printf("Board with valid rows\n");
        for (int i = 0; i < BOARD_SIZE; i++)
        {
            int all[BOARD_SIZE] = {0};
            for (int j = 0; j < BOARD_SIZE; j++)
            {
                if (sudokuBoard[j*BOARD_SIZE + i] > 0)
                    all[sudokuBoard[j*BOARD_SIZE + i] - 1]++;
            }
            for (int j = 0; j < BOARD_SIZE; j++)
                if (all[j] > 1)
                    return false;
        }
        printf("Board with valid columns\n");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int all[BOARD_SIZE] = {0};
                for (int x = i * N; x < (i + 1) * N; x++)
                {
                    for (int y = j * N; y < (j + 1) * N; y++)
                    {
                        if (sudokuBoard[x*BOARD_SIZE + y] > 0)
                    all[sudokuBoard[x*BOARD_SIZE + y] - 1]++;
                    }
                }
                for (int k = 0; k < BOARD_SIZE; k++)
                    if (all[k] > 1)
                    {
                        printf("Invalid block %d:%d\n", i, j);
                        return false;
                    }
            }
        }
        printf("Board with valid blocks\n");
        return true;
}

__host__ void solve(int indx, int *sudokuBoard, int *targetCell, appeared *app, appeared *calculated, const int *start_board, cudaError_t &cudaStatus)
{
    std::queue<int[CELL_COUNT]> Q;
    int currentBoard[CELL_COUNT] = {0};
    int empty_cells[CELL_COUNT] = {-1};
    for (int i = 0; i < CELL_COUNT; i++)
        currentBoard[i] = start_board[i];
    for (int k = 0; k < 10; k++)
    {
        // Calucate notes
        fillEmpty<<<1, indx>>>(sudokuBoard, targetCell, app);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        }
        cudaStatus = cudaMemcpy(calculated, app, CELL_COUNT * sizeof(appeared), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy failed!");
        }

        // Check notes, where we have posibility to enter something in cell

#ifdef DEBUG_MODE
        printf("Current board is "); printf((isBoardValid(currentBoard) ? "valid" : "invalid")); printf(":\n");
        printBoard(currentBoard);
        printBoardInfo(calculated);
#endif

        int iWithLeastOptions = -1;
        int optionsWithI = -1;
        int emptyInAll[CELL_COUNT][BOARD_SIZE] = {0};

        int sureChanges = 0;

        for (int i = 0; i < indx; i++)
        {
            int tmp = 0;
            for (int j = 0; j < BOARD_SIZE; j++)
            {
                if (calculated[i].appeardInBlock[j] == 0 &&
                    calculated[i].appeardInColumn[j] == 0 &&
                    calculated[i].appeardInRow[j] == 0)
                {
                    emptyInAll[i][j] = 1;
                    tmp++;
                }
            }

            if (iWithLeastOptions < 0 || optionsWithI > tmp)
            {
                iWithLeastOptions = i;
                optionsWithI = tmp;
            }
#ifdef DEBUG_MODE
            // printf("[%02d] Cell: %2d; Possible to input: ", i, calculated[i].cell);

            // for (int j = 0; j < BOARD_SIZE; j++)
            //{
            //     printf("%d ", emptyInAll[i][j]);
            // }
            // printf("\n");
#endif
        }

        printf("Least options: %d; For cell: %d\n", optionsWithI, calculated[iWithLeastOptions].cell);
        for (int i = 0; i < BOARD_SIZE; i++)
            if (emptyInAll[iWithLeastOptions][i] == 1)
            {
                currentBoard[calculated[iWithLeastOptions].cell] = i + 1;
                break;
            }

#ifdef DEBUG_MODE
        printf("Board with changes is "); printf((isBoardValid(currentBoard) ? "valid" : "invalid")); printf(":\n");
        printBoard(currentBoard);
#endif

        for (int i = 0; i < CELL_COUNT; i++)
        {
            cudaStatus = cudaMemcpy((sudokuBoard + i * CELL_COUNT), currentBoard, CELL_COUNT * sizeof(int), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "cudaMemcpy failed!");
            }
        }

        indx = 0;
        for (int i = 0; i < CELL_COUNT; i++)
            if (currentBoard[i] == 0)
            {
                empty_cells[indx] = i;
                indx++;
            }

        cudaStatus = cudaMemcpy(targetCell, empty_cells, CELL_COUNT * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy failed!");
        }
    }
}

__global__ void fillEmpty(const int *sudokuBoard, const int *targetCell, appeared *app)
{

    sudokuBoard += threadIdx.x * CELL_COUNT;
    app[threadIdx.x].cell = targetCell[threadIdx.x];

    for (int i = 0; i < BOARD_SIZE; i++)
    {
        app[threadIdx.x].appeardInBlock[i] = app[threadIdx.x].appeardInColumn[i] = app[threadIdx.x].appeardInRow[i] = 0;
    }

    // Calculate notes -- if it turns out there is only one possiblity - insert it

    for (int i = 0; i < BOARD_SIZE; i++)
    {
        int tmp = sudokuBoard[app[threadIdx.x].cell / BOARD_SIZE + i];
        if (tmp > 0)
        {
            app[threadIdx.x].appeardInRow[tmp - 1]++;
        }
    }

    for (int i = 0; i < BOARD_SIZE; i++)
    {

        int tmp = sudokuBoard[app[threadIdx.x].cell % BOARD_SIZE + i * BOARD_SIZE];
        if (tmp > 0)
        {
            app[threadIdx.x].appeardInColumn[tmp - 1]++;
        }
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
        }
    }
#ifdef DEBUG_MODE
    // Check what notes we have

    int numOfZerosRow = 0;
    int numOfZerosColumn = 0;
    int numOfZerosBlock = 0;

    for (int i = 0; i < BOARD_SIZE; i++)
    {
        if (app[threadIdx.x].appeardInRow[i] == 0)
            numOfZerosRow++;
        if (app[threadIdx.x].appeardInColumn[i] == 0)
            numOfZerosColumn++;
        if (app[threadIdx.x].appeardInBlock[i] == 0)
            numOfZerosBlock++;
    }

    printf("TARGET CELL - %d (id: %d); number of zeros row: %d; column: %d; block: %d\n", app[threadIdx.x].cell + 1, threadIdx.x, numOfZerosRow, numOfZerosColumn, numOfZerosBlock);
#endif
    __syncthreads();
}

__host__ int solveSudoku(const int *start_board, int *sudokuBoard, int *targetCell, appeared *app)
{
    cudaError_t cudaStatus;

    int empty_cells[CELL_COUNT] = {-1};

    int indx = 0;
    for (int i = 0; i < CELL_COUNT; i++)
        if (start_board[i] == 0)
        {
            empty_cells[indx] = i;
            indx++;
        }

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

    solve(indx, sudokuBoard, targetCell, app, calculated, start_board, cudaStatus);

Error:
    cudaFree(sudokuBoard);
    cudaFree(targetCell);
    cudaFree(app);

    return cudaStatus;
}

int main()
{

    // const int start_board[CELL_COUNT] =
    //     {
    //         3, 0, 0, 8, 0, 0, 0, 0, 2,
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
            0,
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

    int *sudokuBoard = 0;
    int *targetCell = 0;
    appeared *app = 0;

    return solveSudoku((int *)start_board, sudokuBoard, targetCell, app);
}