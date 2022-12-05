#include "../headers/solver.cuh"

__host__ int solveSudoku(int *start_board)
{
    cudaError_t cudaStatus;
    int *sudokuBoard = 0;

    possibleBoard *poss_d = 0, *poss_h = 0;

    poss_h = new possibleBoard[BOARD_SIZE * NUM_OF_THREADS];

    std::priority_queue<possibleBoard, std::vector<possibleBoard>, decltype(cmpQueue)> S(cmpQueue);

    try
    {

        prepareDevice(cudaStatus, (void **)&poss_d, (void **)&sudokuBoard);

        printBoard(start_board);

        addToQueue(S, start_board, 0);

        // Until there are boards to be checked
        while (!S.empty())
        {
            // Input new boards
            int indx = 0;

#ifdef DEBUG_MODE
            printf("%ld\n", S.size());
#endif

            addNewBoardsToDevice(indx, S, sudokuBoard, cudaStatus);

            // Run Kernels
            runSolver<<<1, indx>>>(sudokuBoard, poss_d);

            // Fetch resoults
            fetchResoults(cudaStatus, poss_h, poss_d);

            // Add new boards to S
            int *tmpBoard;
            if ((tmpBoard = addNewBoardsToQueue(indx, poss_h, S)))
            {
                start_board = tmpBoard;
                break;
            }
        }

        if (isBoardValid(start_board))
        {
            printf("SOLVED!\n");
            printBoard(start_board);
        }
        else
        {
            printf("BOARD UNSOLVABLE!\n");
        }
    }
    catch (sudokuSolverException &e)
    {
        fprintf(stderr, "%s", e.what());
    }

    cudaFree(sudokuBoard);
    cudaFree(poss_d);
    delete[] poss_h;

    return cudaStatus;
}

__host__ void prepareDevice(cudaError_t &cudaStatus, void **poss_d, void **sudokuBoard) throw()
{
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        std::ostringstream errMess;
        errMess << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed? Returned error code " << cudaStatus;
        throw sudokuSolverException(errMess.str());
    }

    cudaStatus = cudaMalloc(sudokuBoard, NUM_OF_THREADS * CELL_COUNT * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        std::ostringstream errMess;
        errMess << "cudaMalloc failed! Returned error code " << cudaStatus;
        throw sudokuSolverException(errMess.str());
    }

    cudaStatus = cudaMalloc(poss_d, NUM_OF_THREADS * BOARD_SIZE * sizeof(possibleBoard));
    if (cudaStatus != cudaSuccess)
    {
        std::ostringstream errMess;
        errMess << "cudaMalloc failed! Returned error code " << cudaStatus;
        throw sudokuSolverException(errMess.str());
    }
}

__host__ void addNewBoardsToDevice(int &indx, std::priority_queue<possibleBoard, std::vector<possibleBoard>, decltype(cmpQueue)> &Q, int *sudokuBoard, cudaError_t &cudaStatus) throw()
{
    int tmpSudokuBoard[CELL_COUNT * NUM_OF_THREADS];

    // Fetch as many boards as we have Threads avaliable
    for (; indx < NUM_OF_THREADS && !Q.empty(); indx++)
    {
        possibleBoard tmp = Q.top();
        Q.pop();
#ifdef DEBUG_MODE
        printf("Running thread %02d with board:\n", indx + 1);
        printBoard(tmp.board);
#endif
        // Write down the boards to GPU memory
        for (int j = 0; j < CELL_COUNT; j++)
        {
            tmpSudokuBoard[indx * CELL_COUNT + j] = tmp.board[j];
        }
    }

    // Copy memory and run kernel
    cudaStatus = cudaMemcpy(sudokuBoard, tmpSudokuBoard, (NUM_OF_THREADS)*CELL_COUNT * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        std::ostringstream errMess;
        errMess << "cudaMemcpy failed! Returned error code " << cudaStatus;
        throw sudokuSolverException(errMess.str());
    }

#ifdef DEBUG_MODE
    printf("Running %02d threads\n", indx);
#endif
}

__host__ int *addNewBoardsToQueue(int &indx, possibleBoard *poss_h, std::priority_queue<possibleBoard, std::vector<possibleBoard>, decltype(cmpQueue)> &Q)
{
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
                    return poss_h[i * BOARD_SIZE + j].board;
                }

                addToQueue(Q, poss_h[i * BOARD_SIZE + j].board, poss_h[i * BOARD_SIZE + j].status);
            }
        }
    }

    return nullptr;
}
__host__ void addToQueue(std::priority_queue<possibleBoard, std::vector<possibleBoard>, decltype(cmpQueue)> &Q, int *board, int status)
{
    possibleBoard tmp;
    for (int i = 0; i < CELL_COUNT; i++)
        tmp.board[i] = board[i];
    tmp.status = 0;
    Q.push(tmp);
}
__host__ void fetchResoults(cudaError_t &cudaStatus, possibleBoard *poss_h, possibleBoard *poss_d) throw()
{
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        std::ostringstream errMess;
        errMess << "Kernel launch failed: " << cudaGetErrorString(cudaStatus);
        throw sudokuSolverException(errMess.str());
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        std::ostringstream errMess;
        errMess << "cudaDeviceSynchronize returned error code " << cudaStatus;
        throw sudokuSolverException(errMess.str());
    }

    cudaStatus = cudaMemcpy(poss_h, poss_d, NUM_OF_THREADS * BOARD_SIZE * sizeof(possibleBoard), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        std::ostringstream errMess;
        errMess << "cudaMemcpy failed! Returned error code " << cudaStatus;
        throw sudokuSolverException(errMess.str());
    }
}