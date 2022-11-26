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

//#define DEBUG_MODE

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
__host__ __device__ bool isBoardValid(int *sudokuBoard)
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
__device__ bool isBoardCorrect(const int *sudokuBoard)
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
            if (all[j] > 1)
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
            if (all[j] > 1)
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
                if (all[k] > 1)
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
            emptyInAll[i] = appeardInBlock[i] = appeardInColumn[i] = appeardInRow[i] = 0;
        }
        tmp = 0;
    }

    *possCount = indx;
}

__global__ void runSolver(const int *currentBoard, possibleBoard *possBoard)
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

    int numOfEmptyCells = 0;

    currentBoard += (CELL_COUNT)*threadIdx.x;
    possBoard += (BOARD_SIZE)*threadIdx.x;
    if (!isBoardCorrect(currentBoard))
        return;

    possibilitie *poss = new possibilitie[CELL_COUNT];
    for (int i = 0; i < CELL_COUNT; i++)
    {
        if (currentBoard[i] == 0)
        {
            emptyCells[indx] = i;
            indx++;
        }
    }
    for (int i = 0; i < BOARD_SIZE; i++)
    {
        for (int j = 0; j < CELL_COUNT; j++)
            possBoard[i].board[j] = currentBoard[j];

        possBoard[i].status = 0;
    }
    for (int i = 0; i < CELL_COUNT; i++)
    {
        poss[i].cell = 0;
        for (int j = 0; j < BOARD_SIZE; j++)
            poss[i].poss[j] = 0;
    }

    numOfEmptyCells = indx;
    calculatePossibilites(currentBoard, (int *)emptyCells, poss, &indx);

    // We now have all possible otions that can be safely inputted into our
    // current board.

    int leastOption = 11, iWithLeastOptions = -1;

    for (int i = 0; i < indx; i++)
    {
        int tmp = 0;
        for (int j = 0; j < BOARD_SIZE; j++)
            tmp += poss[i].poss[j];
        if (tmp < leastOption && tmp > 0)
        {
            leastOption = tmp;
            iWithLeastOptions = i;
        }
    }

    int countOfBoards = 0;
    if (iWithLeastOptions > -1)
        for (int i = 0; i < BOARD_SIZE; i++)
        {
            if (poss[iWithLeastOptions].poss[i] == 1)
            {

                possBoard[countOfBoards].status = leastOption + numOfEmptyCells;
                possBoard[countOfBoards].board[poss[iWithLeastOptions].cell] = i + 1;
                countOfBoards++;
            }
        }

    delete[] poss;
}
__host__ int solveSudoku(int *start_board)
{
    cudaError_t cudaStatus;
    int *sudokuBoard = 0;
    int tmpSudokuBoard[CELL_COUNT * NUM_OF_THREADS];
    possibleBoard *poss_d = 0, *poss_h = 0;

    poss_h = new possibleBoard[BOARD_SIZE * NUM_OF_THREADS];
    auto cmp = [](possibleBoard left, possibleBoard right)
    { return (left.status) > (right.status); };
    std::priority_queue<possibleBoard, std::vector<possibleBoard>, decltype(cmp)> S(cmp);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void **)&sudokuBoard, NUM_OF_THREADS * CELL_COUNT * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void **)&poss_d, NUM_OF_THREADS * BOARD_SIZE * sizeof(possibleBoard));
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

    printBoard((int *)start_board);
    runSolver<<<1, 1>>>(sudokuBoard, poss_d);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(poss_h, poss_d, NUM_OF_THREADS * BOARD_SIZE * sizeof(possibleBoard), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed! Returned error code %d\n", cudaStatus);
        goto Error;
    }

    for (int i = 0; i < NUM_OF_THREADS; i++)
    {
        for (int j = 0; j < BOARD_SIZE; j++)
        {
            if (poss_h[i * BOARD_SIZE + j].status >= 1)
            {
#ifdef DEBUG_MODE
                printf("Possible board [THREAD: %d][POSS: %d]:\n", i + 1, j + 1);
                printBoard(poss_h[i * BOARD_SIZE + j].board);
#endif
                possibleBoard tmp;
                for (int k = 0; k < CELL_COUNT; k++)
                    tmp.board[k] = poss_h[i * BOARD_SIZE + j].board[k];
                tmp.status = poss_h[i * BOARD_SIZE + j].status;
                S.push(tmp);
            }
        }
    }

    while (!S.empty())
    {
        // Input new boards
        int indx = 0;
#ifdef DEBUG_MODE
        printf("%ld\n", S.size());
#endif
        for (; indx < NUM_OF_THREADS && !S.empty(); indx++)
        {
            possibleBoard tmp = S.top();
            S.pop();
#ifdef DEBUG_MODE
            printf("Running thread %02d with board:\n", indx + 1);
            printBoard(tmp.board);
#endif
            for (int j = 0; j < CELL_COUNT; j++)
            {
                tmpSudokuBoard[indx * CELL_COUNT + j] = tmp.board[j];
            }
        }

        // Copy memory and run kernel
        cudaStatus = cudaMemcpy(sudokuBoard, tmpSudokuBoard, (NUM_OF_THREADS)*CELL_COUNT * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
#ifdef DEBUG_MODE
        printf("Running %02d threads\n", indx);
#endif
        runSolver<<<1, indx>>>(sudokuBoard, poss_d);

        // Fetch resoults
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
            goto Error;
        }

        cudaStatus = cudaMemcpy(poss_h, poss_d, NUM_OF_THREADS * BOARD_SIZE * sizeof(possibleBoard), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy failed! Returned error code %d\n", cudaStatus);
            goto Error;
        }

        // Add new boards to S
        for (int i = 0; i < indx; i++)
        {
            for (int j = 0; j < BOARD_SIZE; j++)
            {
                if (poss_h[i * BOARD_SIZE + j].status >= 1)
                {
#ifdef DEBUG_MODE
                    printf("Possible board [THREAD: %d][POSS: %d]:\n", i + 1, j + 1);
                    printBoard(poss_h[i * BOARD_SIZE + j].board);
#endif
                    if (isBoardValid(poss_h[i * BOARD_SIZE + j].board))
                    {
                        printf("SOLVED!\n");
                        printBoard(poss_h[i * BOARD_SIZE + j].board);
                        goto Error;
                    }

                    possibleBoard tmp;
                    for (int k = 0; k < CELL_COUNT; k++)
                        tmp.board[k] = poss_h[i * BOARD_SIZE + j].board[k];
                    tmp.status = poss_h[i * BOARD_SIZE + j].status;
                    S.push(tmp);
                }
            }
        }
    }

Error:
    cudaFree(sudokuBoard);
    cudaFree(poss_d);
    delete[] poss_h;

    return cudaStatus;
}

__host__ void loadBoard(int *board, std::ifstream &inFile)
{
    std::string line;
    for (int i = 0; i < BOARD_SIZE && getline(inFile, line); i++)
    {
        if (line.length() != BOARD_SIZE)
        {
            i--;
        }
        else
            for (int j = 0; j < BOARD_SIZE; j++)
            {
                board[i * BOARD_SIZE + j] = line[j] - '0';
            }
    }
}

int main(int argc, char **argv)
{

    std::ifstream inFS;
    std::string fileName;

    std::cout << "Input file: ";
    std::cin >> fileName;

    int *board = new int[CELL_COUNT];

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    int tmp;

    inFS.open(fileName);

    while (!inFS.eof())
    {
        loadBoard(board, inFS);
        begin = std::chrono::steady_clock::now();
        tmp = solveSudoku(board);
        end = std::chrono::steady_clock::now();
        if (tmp != 0)
            return tmp;
        std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    }

    inFS.close();
    delete[] board;

    return tmp;
}