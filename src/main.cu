#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <stack>
#include <chrono>

#define N 3
#define BOARD_SIZE (N * N)
#define CELL_COUNT (BOARD_SIZE * BOARD_SIZE)

// #define DEBUG_MODE

struct appeared
{
    int cell = -1;

    int appeardInRow[BOARD_SIZE] = {0};
    int appeardInColumn[BOARD_SIZE] = {0};
    int appeardInBlock[BOARD_SIZE] = {0};

} typedef appeared;

struct possibilitie
{
    int cell = -1;
    int poss[BOARD_SIZE] = {0};

} typedef possibilitie;

struct move
{
    int board[CELL_COUNT] = {0};
    int cell = -1;
    int possibilites[BOARD_SIZE] = {0};

    move(int *currentBoard, int cell, int *possibilites)
    {
        for (int i = 0; i < CELL_COUNT; i++)
            this->board[i] = currentBoard[i];

        for (int i = 0; i < BOARD_SIZE; i++)
            this->possibilites[i] = possibilites[i];

        this->cell = cell;
    }
} typedef move;

__global__ void
fillEmpty(const int *sudokuBoard, const int *targetCell, appeared *app);

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
            if (sudokuBoard[i * BOARD_SIZE + j] > 0)
                all[sudokuBoard[i * BOARD_SIZE + j] - 1]++;
        }
        for (int j = 0; j < BOARD_SIZE; j++)
            if (all[j] != 1)
                return false;
    }
    for (int i = 0; i < BOARD_SIZE; i++)
    {
        int all[BOARD_SIZE] = {0};
        for (int j = 0; j < BOARD_SIZE; j++)
        {
            if (sudokuBoard[j * BOARD_SIZE + i] > 0)
                all[sudokuBoard[j * BOARD_SIZE + i] - 1]++;
        }
        for (int j = 0; j < BOARD_SIZE; j++)
            if (all[j] != 1)
                return false;
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int all[BOARD_SIZE] = {0};
            for (int x = i * N; x < (i + 1) * N; x++)
            {
                for (int y = j * N; y < (j + 1) * N; y++)
                {
                    if (sudokuBoard[x * BOARD_SIZE + y] > 0)
                        all[sudokuBoard[x * BOARD_SIZE + y] - 1]++;
                }
            }
            for (int k = 0; k < BOARD_SIZE; k++)
                if (all[k] != 1)
                {
                    return false;
                }
        }
    }
    return true;
}

__device__ void calculatePossibilites(const int *currentBoard, int *emptyCells, possibilitie *poss, int *possCount)
{
    int cell = -1;

    int appeardInRow[BOARD_SIZE] = {0};
    int appeardInColumn[BOARD_SIZE] = {0};
    int appeardInBlock[BOARD_SIZE] = {0};
    int emptyInAll[BOARD_SIZE] = {0};

    int indx = 0;
    int tmp = 0;

    for (int k = 0; k < *possCount; k++)
    {
        cell = emptyCells[k];

        for (int i = 0; i < BOARD_SIZE; i++)
        {
            int tmp = currentBoard[(cell / BOARD_SIZE) * BOARD_SIZE + i];
            if (tmp > 0)
            {
                appeardInRow[tmp - 1]++;
            }
        }

        for (int i = 0; i < BOARD_SIZE; i++)
        {

            int tmp = currentBoard[cell % BOARD_SIZE + i * BOARD_SIZE];
            if (tmp > 0)
            {
                appeardInColumn[tmp - 1]++;
            }
        }

        int firstCellOfBlock = ((cell / BOARD_SIZE) / N) * BOARD_SIZE * N + ((cell % BOARD_SIZE) / N) * N;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int tmp = currentBoard[firstCellOfBlock + i * BOARD_SIZE + j];
                if (tmp > 0)
                {
                    appeardInBlock[tmp - 1]++;
                }
            }
        }

        // Remember the possibilites
        for (int j = 0; j < BOARD_SIZE; j++)
        {
            if (appeardInBlock[j] == 0 &&
                appeardInColumn[j] == 0 &&
                appeardInRow[j] == 0)
            {
                emptyInAll[j] = 1;
                tmp++;
            }
        }
        if (tmp > 0)
        {
            for (int i = 0; i < BOARD_SIZE; i++)
                poss[indx].poss[i] = emptyInAll[i];
            poss[indx].cell = cell;
            indx++;
        }

        // Reset arrays
        for (int i = 0; i < BOARD_SIZE; i++)
        {
            appeardInBlock[i] = appeardInColumn[i] = appeardInRow[i] = 0;
        }
        tmp = 0;
    }

    *possCount = indx;
}

__global__ void runSolver(const int *currentBoard)
{
    // Kerenl recives a board as an array size of CELL_COUNT
    // It generates valid boards that can be created
    // using given array and returns it to host.
    // If in board exists cells that are "sure" meaning only
    // one number can be inputed there, we only consider those
    // boards.

    // Count how many empty cells we have
    // i.e. how many possibilites can ther be
    int indx = 0;
    int emptyCells[CELL_COUNT] = {0};
    for (int i = 0; i < CELL_COUNT; i++)
    {
        if (currentBoard[i] == 0)
        {
            emptyCells[indx] = i;
            indx++;
        }
    }

    possibilitie *poss = new possibilitie[indx];

    calculatePossibilites(currentBoard, (int *)emptyCells, poss, &indx);

    // We now have all possible otions that can be safely inputted into our
    // current board.
    for (int i = 0; i < indx; i++)
    {
        int tmp = 0;
        for (int j = 0; j < BOARD_SIZE; j++)
            if (poss[i].poss[j] == 1)
                tmp++;
    }

    delete[] poss;
}

__host__ void solve(int indx, int *sudokuBoard, int *targetCell, appeared *app, appeared *calculated, const int *start_board, cudaError_t &cudaStatus)
{
    std::stack<move> S;

    int currentBoard[CELL_COUNT] = {0};
    int empty_cells[CELL_COUNT] = {-1};
    for (int i = 0; i < CELL_COUNT; i++)
        currentBoard[i] = start_board[i];

    printBoard(currentBoard);
    while (!isBoardValid(currentBoard))
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
        printf("Current board is ");
        printf((isBoardValid(currentBoard) ? "valid" : "invalid"));
        printf(":\n");
        printBoard(currentBoard);
        // printBoardInfo(calculated);
#endif

        int iWithLeastOptions = -1;
        int optionsWithI = -1;
        int emptyInAll[CELL_COUNT][BOARD_SIZE] = {0};

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

            if ((iWithLeastOptions < 0 || optionsWithI > tmp) && tmp > 0)
            {
                iWithLeastOptions = i;
                optionsWithI = tmp;
            }
        }

// Do move
#ifdef DEBUG_MODE
        printf("Least options: %d; For cell: %d\n", optionsWithI, calculated[iWithLeastOptions].cell);
#endif
        if (optionsWithI == 1)
        {
            for (int i = 0; i < BOARD_SIZE; i++)
                if (emptyInAll[iWithLeastOptions][i] == 1)
                {
                    emptyInAll[iWithLeastOptions][i] = 0;
                    move m = move((int *)currentBoard, calculated[iWithLeastOptions].cell, (int *)emptyInAll[iWithLeastOptions]);
                    currentBoard[calculated[iWithLeastOptions].cell] = i + 1;
                    S.push(m);
                    break;
                }
        }
        else if (optionsWithI > 1)
        {
            // Input random number and check if board is valid
            for (int i = 0; i < BOARD_SIZE; i++)
            {
                if (emptyInAll[iWithLeastOptions][i] == 1)
                {
                    emptyInAll[iWithLeastOptions][i] = 0;
                    currentBoard[calculated[iWithLeastOptions].cell] = i + 1;
                    move m = move((int *)currentBoard, calculated[iWithLeastOptions].cell, (int *)emptyInAll[iWithLeastOptions]);
                    S.push(m);
                    break;
                }
            }
        }
        else if (!isBoardValid(currentBoard))
        {
// Board is broken, we need to back up
#ifdef DEBUG_MODE
            printf("We are backing up\n");
#endif
            bool foundMove = false;
            while (!S.empty() && !foundMove)
            {
                move m = S.top();
                S.pop();
                int numOfPossibilites = 0;
                for (int i = 0; i < BOARD_SIZE; i++)
                    numOfPossibilites += m.possibilites[i];

                // We do not have an option to diverge from this state
                if (numOfPossibilites == 0)
                {
#ifdef DEBUG_MODE
                    printf("Move with no options\n");
#endif
                    continue;
                }

                for (int i = 0; i < BOARD_SIZE; i++)
                {
                    if (m.possibilites[i] == 1)
                    {
                        // We have a possiblitie to diverge
                        m.possibilites[i] = 0;
                        for (int j = 0; j < CELL_COUNT; j++)
                            currentBoard[j] = m.board[j];
                        currentBoard[m.cell] = i + 1;
                        S.push(m);
#ifdef DEBUG_MODE
                        printf("Move with   options - cell: %d set to: %d\n", m.cell, i + 1);
#endif
                        foundMove = true;
                        break;
                    }
                }
            }
            if (S.empty())
            {
                printf("No solution :(\n");
                return;
            }
        }

        // Prepear next step

#ifdef DEBUG_MODE
        printf("Board with changes is ");
        printf((isBoardValid(currentBoard) ? "valid" : "invalid"));
        printf(":\n");
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
    printf("Solved!\n");
    printBoard(currentBoard);
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
        int tmp = sudokuBoard[(app[threadIdx.x].cell / BOARD_SIZE) * BOARD_SIZE + i];
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
    printBoard((int *)start_board);
    runSolver<<<1, 1>>>(sudokuBoard);

Error:
    cudaFree(sudokuBoard);
    cudaFree(targetCell);
    cudaFree(app);

    return cudaStatus;
}

int main()
{
    // Easy
    // const int start_board[CELL_COUNT] =
    //     {
    //         3, 8, 6, 0, 0, 4, 7, 0, 0,
    //         0, 0, 9, 0, 0, 0, 2, 0, 0,
    //         0, 2, 0, 1, 0, 3, 8, 0, 5,
    //         0, 7, 8, 0, 3, 0, 6, 2, 0,
    //         0, 5, 2, 0, 0, 1, 0, 0, 4,
    //         9, 4, 0, 2, 7, 0, 0, 0, 0,
    //         2, 3, 0, 7, 4, 9, 5, 8, 6,
    //         8, 0, 0, 0, 1, 0, 4, 0, 0,
    //         4, 0, 0, 0, 0, 0, 0, 0, 2,
    //     };
    // Hard
    // const int start_board[CELL_COUNT] =
    //     {
    //         0, 0, 1, 0, 0, 0, 3, 6, 0,
    //         0, 0, 0, 0, 2, 0, 0, 0, 0,
    //         3, 0, 0, 0, 5, 6, 0, 8, 0,
    //         0, 0, 0, 9, 0, 0, 0, 0, 0,
    //         0, 4, 0, 0, 0, 0, 0, 0, 7,
    //         1, 0, 0, 0, 3, 8, 0, 5, 0,
    //         0, 0, 0, 1, 0, 0, 0, 9, 0,
    //         0, 0, 7, 0, 6, 9, 0, 0, 5,
    //         6, 0, 0, 2, 0, 0, 0, 0, 0,
    //     };
    // const int start_board[CELL_COUNT] =
    //     {
    //         0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     };

    const int start_board[CELL_COUNT] =
        {
            0,
            0,
            1,
            0,
            0,
            0,
            3,
            6,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
            3,
            0,
            0,
            0,
            5,
            6,
            0,
            8,
            0,
            0,
            0,
            0,
            9,
            0,
            0,
            0,
            0,
            0,
            0,
            4,
            0,
            0,
            0,
            0,
            0,
            0,
            7,
            1,
            0,
            0,
            0,
            3,
            8,
            0,
            5,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            9,
            0,
            0,
            0,
            7,
            0,
            6,
            9,
            0,
            0,
            5,
            6,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
        };

    int *sudokuBoard = 0;
    int *targetCell = 0;
    appeared *app = 0;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    int tmp = solveSudoku((int *)start_board, sudokuBoard, targetCell, app);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    return tmp;
}