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