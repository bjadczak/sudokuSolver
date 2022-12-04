#include <iostream>
#include <exception>

class sudokuSolverException : public std::exception
{
private:
    std::string message;

public:
    virtual const char *what() const throw();
    sudokuSolverException(std::string message);
};