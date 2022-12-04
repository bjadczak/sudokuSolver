#include "../headers/kernel.cuh"

__device__ void calculatePossibilites(const int *currentBoard, int *emptyCells, possibilitie *poss, int *possCount)
{
    int cell = -1;

    int appeardInRow[BOARD_SIZE] = {0};
    int appeardInColumn[BOARD_SIZE] = {0};
    int appeardInBlock[BOARD_SIZE] = {0};
    int emptyInAll[BOARD_SIZE] = {0};

    int indx = 0;
    int tmp = 0;

    // Look thourgh all empty cells

    for (int k = 0; k < *possCount; k++)
    {
        cell = emptyCells[k];

        // Based on what is in other cells we input possible numbers
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
        // We ONLY store options that have possiblitie to be inputed
        // If cell is locked i.e. there is no possiblitie for anything
        // to be inputted there we do not consider it
        if (tmp > 0)
        {
            for (int i = 0; i < BOARD_SIZE; i++)
                poss[indx].poss[i] = emptyInAll[i];
            poss[indx].cell = cell;
            indx++;
        }
        else
        {
            // Discard this Kernels board
            *possCount = 0;
            indx = 0;
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

    // We now have all possible options that can be safely inputted into our
    // current board. We look through them to find cell with least possibilities
    // then we generate boards with all possibilites for that cell.
    //
    // We also store how many possibilites there were and how many empty cells we
    // had in this one board.

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