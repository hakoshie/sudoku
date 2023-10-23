#!/usr/bin/env python3
# https://www.cs.mcgill.ca/~aassaf9/python/algorithm_x.html
# Author: Ali Assaf <ali.assaf.mail@gmail.com>
# Copyright: (C) 2010 Ali Assaf
# License: GNU General Public License <http://www.gnu.org/licenses/>
from itertools import product

def solve_sudoku(size, grid):
    """ An efficient Sudoku solver using Algorithm X.

    >>> grid = [
    ...     [5, 3, 0, 0, 7, 0, 0, 0, 0],
    ...     [6, 0, 0, 1, 9, 5, 0, 0, 0],
    ...     [0, 9, 8, 0, 0, 0, 0, 6, 0],
    ...     [8, 0, 0, 0, 6, 0, 0, 0, 3],
    ...     [4, 0, 0, 8, 0, 3, 0, 0, 1],
    ...     [7, 0, 0, 0, 2, 0, 0, 0, 6],
    ...     [0, 6, 0, 0, 0, 0, 2, 8, 0],
    ...     [0, 0, 0, 4, 1, 9, 0, 0, 5],
    ...     [0, 0, 0, 0, 8, 0, 0, 7, 9]]
    >>> for solution in solve_sudoku((3, 3), grid):
    ...     print(*solution, sep='\\n')
    [5, 3, 4, 6, 7, 8, 9, 1, 2]
    [6, 7, 2, 1, 9, 5, 3, 4, 8]
    [1, 9, 8, 3, 4, 2, 5, 6, 7]
    [8, 5, 9, 7, 6, 1, 4, 2, 3]
    [4, 2, 6, 8, 5, 3, 7, 9, 1]
    [7, 1, 3, 9, 2, 4, 8, 5, 6]
    [9, 6, 1, 5, 3, 7, 2, 8, 4]
    [2, 8, 7, 4, 1, 9, 6, 3, 5]
    [3, 4, 5, 2, 8, 6, 1, 7, 9]
    """
    R, C = size
    N = R * C
    target = ([("rc", rc) for rc in product(range(N), range(N))] +
         [("rn", rn) for rn in product(range(N), range(1, N + 1))] +
         [("cn", cn) for cn in product(range(N), range(1, N + 1))] +
         [("bn", bn) for bn in product(range(N), range(1, N + 1))])
    covers = dict()
    for row, col, n in product(range(N), range(N), range(1, N + 1)):
        b = (row // R) * R + (col // C) # Box number
        covers[(row, col, n)] = [
            ("rc", (row, col)),
            ("rn", (row, n)),
            ("cn", (col, n)),
            ("bn", (b, n))]
    target, covers = exact_cover(target, covers)
    for i, row in enumerate(grid):
        for j, n in enumerate(row):
            if n:
                select(target, covers, (i, j, n))
    for solution in solve(target, covers, []):
        for (row, col, n) in solution:
            grid[row][col] = n
        yield grid

def exact_cover(target, covers):
    target = {j: set() for j in target}
    for i, row in covers.items():
        for j in row:
            target[j].add(i)
    return target, covers

def solve(target, covers, solution):
    if not target:
        yield list(solution)
    else:
        col = min(target, key=lambda c: len(target[c]))
        for row in list(target[col]):
            solution.append(row)
            cols = select(target, covers, row)
            for s in solve(target, covers, solution):
                yield s
            deselect(target, covers, row, cols)
            solution.pop()

def select(target, covers, row):
    cols = []
    for j in covers[row]:
        for i in target[j]:
            for k in covers[i]:
                if k != j:
                    target[k].remove(i)
        cols.append(target.pop(j))
    return cols

def deselect(target, covers, row, cols):
    for j in reversed(covers[row]):
        target[j] = cols.pop()
        for i in target[j]:
            for k in covers[i]:
                if k != j:
                    target[k].add(i)

if __name__ == "__main__":
    import doctest
    doctest.testmod()