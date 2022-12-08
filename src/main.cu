#include "headers/solver.cuh"

#include <stdio.h>
#include <iostream>
#include <fstream>

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

    int timeCountTotal = 0;
    int boardsCount = 0;
    int solvedBoards = 0;
    bool ifSolved;

    inFS.open(fileName);

    while (!inFS.eof())
    {
        loadBoard(board, inFS);
        begin = std::chrono::steady_clock::now();
        tmp = solveSudoku(board, &ifSolved);
        end = std::chrono::steady_clock::now();
        if (tmp != 0)
            return tmp;
        tmp = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "Time elapsed = " << tmp << "[ms]" << std::endl;
        timeCountTotal += tmp;
        boardsCount++;
        if (ifSolved)
            solvedBoards++;
    }

    std::cout << "Total time elapsed = " << timeCountTotal << "[ms]" << std::endl
              << "Boards (solved) = " << boardsCount << "(" << solvedBoards << ")" << std::endl
              << "Average time per board = " << (double)timeCountTotal / (double)boardsCount << "[ms]" << std::endl;
    ;

    inFS.close();
    delete[] board;

    return tmp;
}