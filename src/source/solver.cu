#include "../headers/solver.cuh"

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

    // We run 1 Kernel for our start board, this will give us at least 1 (if board is not finished on start)
    // up to 9 boards. For those we run the solver further
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

    // Store the possible boards in priority queue
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

    // Until there are boards to be checked
    while (!S.empty())
    {
        // Input new boards
        int indx = 0;
#ifdef DEBUG_MODE
        printf("%ld\n", S.size());
#endif
        // Fetch as many boards as we have Threads avaliable
        for (; indx < NUM_OF_THREADS && !S.empty(); indx++)
        {
            possibleBoard tmp = S.top();
            S.pop();
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
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
#ifdef DEBUG_MODE
        printf("Running %02d threads\n", indx);
#endif
        // Run Kernels
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

    if (isBoardValid(start_board))
    {
        printf("SOLVED!\n");
        printBoard(start_board);
    }
    else
    {
        printf("BOARD UNSOLVABLE!\n");
    }

Error:
    cudaFree(sudokuBoard);
    cudaFree(poss_d);
    delete[] poss_h;

    return cudaStatus;
}
