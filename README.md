# sudokuSolver

This is an Univeristy project for *Graphic Processors in Computational Applications* course on Computer Scince course.

---

Project is an implementation of **Cross-hatching** alghorithm using CUDA C++. Main notion of alghorithm is checking what can be written in given cell, by considering whole row, column, and block. Then we generate new boards with that information. With generated boards we are using modification of backtracking alghorithm to find what might be a solution.

## Usage

After building program with `makefile` provided we need `.txt` file with boards. There is an example file `boards.txt`. Syntax is simple:
- 0 is considered as empty cell
- separete boards with one empty line

Example file looks like this:
```
386004700
009000200
020103805
078030620
052001004
940270000
230749586
800010400
400000002

001000360
000020000
300056080
000900000
040000007
100038050
000100090
007069005
600200000

001000360
000020000
300056080
000900000
040000007
100038050
000100090
007069005
600200007
```

## Output

In output we can see printed board, and message `SOLVED!` or `BOARD UNSOLVABLE!`. If board was solved, we also will see printeded solved board. After that we can see how much time alghorithm took. Example output looks like this:
```
3 8 6 0 0 4 7 0 0 
0 0 9 0 0 0 2 0 0 
0 2 0 1 0 3 8 0 5 
0 7 8 0 3 0 6 2 0 
0 5 2 0 0 1 0 0 4 
9 4 0 2 7 0 0 0 0 
2 3 0 7 4 9 5 8 6 
8 0 0 0 1 0 4 0 0 
4 0 0 0 0 0 0 0 2 
SOLVED!
3 8 6 5 2 4 7 9 1 
5 1 9 8 6 7 2 4 3 
7 2 4 1 9 3 8 6 5 
1 7 8 4 3 5 6 2 9 
6 5 2 9 8 1 3 7 4 
9 4 3 2 7 6 1 5 8 
2 3 1 7 4 9 5 8 6 
8 9 5 6 1 2 4 3 7 
4 6 7 3 5 8 9 1 2 
Time elapsed = 83[ms]
0 0 1 0 0 0 3 6 0 
0 0 0 0 2 0 0 0 0 
3 0 0 0 5 6 0 8 0 
0 0 0 9 0 0 0 0 0 
0 4 0 0 0 0 0 0 7 
1 0 0 0 3 8 0 5 0 
0 0 0 1 0 0 0 9 0 
0 0 7 0 6 9 0 0 5 
6 0 0 2 0 0 0 0 0 
SOLVED!
7 5 1 8 9 4 3 6 2 
8 9 6 3 2 1 5 7 4 
3 2 4 7 5 6 9 8 1 
5 6 3 9 4 7 1 2 8 
9 4 8 5 1 2 6 3 7 
1 7 2 6 3 8 4 5 9 
4 8 5 1 7 3 2 9 6 
2 3 7 4 6 9 8 1 5 
6 1 9 2 8 5 7 4 3 
Time elapsed = 129[ms]
0 0 1 0 0 0 3 6 0 
0 0 0 0 2 0 0 0 0 
3 0 0 0 5 6 0 8 0 
0 0 0 9 0 0 0 0 0 
0 4 0 0 0 0 0 0 7 
1 0 0 0 3 8 0 5 0 
0 0 0 1 0 0 0 9 0 
0 0 7 0 6 9 0 0 5 
6 0 0 2 0 0 0 0 7 
BOARD UNSOLVABLE!
Time elapsed = 76[ms]
```

At the end we get summary of how much time it took for whole file, what is the average time for solving board, and how many boards were processed (and how many where successfully solved). For example:
```
Total time elapsed = 7464[ms]
Boards (solved) = 55(54)
Average time per board = 135.709[ms]
```

---
Last 50 sudoku boards in boards.txt are from https://projecteuler.net 
