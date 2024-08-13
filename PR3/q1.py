import numpy as np

def binary_matrix_rank(matrix):
    num_rows, num_cols = matrix.shape

    rank = 0
    for col in range(num_cols):
        pivot_row = rank

        while pivot_row < num_rows and matrix[pivot_row][col] == 0:
            pivot_row += 1

        if pivot_row == num_rows:

            continue

        matrix[pivot_row] = [1 - entry for entry in matrix[pivot_row]]

        for i in range(rank + 1, num_rows):
            if matrix[i][col] == 1:
                matrix[i] = [entry ^ matrix[pivot_row][j] for j, entry in enumerate(matrix[i])]

        rank += 1

    return rank

def find_rank(matrix):
    num_rows, num_cols = matrix.shape
    rank = 0

    for col in range(num_cols):
        pivot_row = None

        for row in range(rank, num_rows):
            if matrix[row, col] == 1:
                pivot_row = row
                break

        if pivot_row is not None:
            matrix[[rank, pivot_row]] = matrix[[pivot_row, rank]]

            for i in range(num_rows):
                if i != rank and matrix[i, col] == 1:
                    matrix[i] ^= matrix[rank]

            rank += 1

    return rank

def allZero(matrix):
    num_rows, num_cols = matrix.shape
    for i in range (num_rows):
        for j in range (num_cols):
            if (matrix[i][j] != 0): return False
    return True

rows, cols = map(int, input().split())
matrix = np.zeros((rows, cols), dtype=int)

for i in range(rows):
    row_input = input()
    row = [int(bit) for bit in row_input]
    matrix[i, :] = [int(bit) for bit in row_input]

rank = find_rank(matrix)

if (allZero(matrix)): print(0)
elif (rank == 0): print(rows*cols)
else: print(rows*cols - rank + 1)
