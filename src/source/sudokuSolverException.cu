#include "../headers/sudokuSolverException.cuh"

const char *sudokuSolverException::what() const throw()
{
    return message.c_str();
}
sudokuSolverException::sudokuSolverException(std::string message)
{
    this->message = message;
}