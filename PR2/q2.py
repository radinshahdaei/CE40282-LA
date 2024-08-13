import numpy as np
import sys

EPSILON = 10 ** (-12)

def inputVectors(n, m):
    matrix = np.zeros((m, n))

    for i in range(n):
        vector = input().strip().split()
        matrix[:, i] = np.array(vector, dtype=float)

    return matrix

def createMatrix(matrix, n, m):
    new_matrix = np.zeros((m, n - 1))
    for i in range(1, n):
        new_matrix[:, i - 1] = matrix[:, i] - matrix[:, 0]
        matrix[np.abs(matrix) < EPSILON] = 0
    return new_matrix

def createPivot(index, matrix):
    matrix[np.abs(matrix) < EPSILON] = 0
    n = min(matrix.shape[1],matrix.shape[0])
    flag = False
    if matrix[index][index] == 0:
        for i in range (index + 1, n):
            if (matrix[i][index] != 0):
                matrix[[index, i]] = matrix[[i, index]]
                flag = True
                break
    else:
        flag = True
    if flag == False:
        return 0
    for i in range (index + 1, n):
        mult = matrix[i][index] / matrix[index][index]
        matrix[np.abs(matrix) < EPSILON] = 0    
        matrix[i] -= mult * matrix[index]
        matrix[np.abs(matrix) < EPSILON] = 0
    return 1

def createPivots(matrix):
    matrix[np.abs(matrix) < EPSILON] = 0
    n = min(matrix.shape[0],matrix.shape[1])
    for i in range(n):
        answer = createPivot(i, matrix)
        if (answer == 0):
            return 0
    return 1

input_str = input()
n, m = map(int, input_str.split())
matrix = inputVectors(n, m)
new_matrix = np.copy(matrix)

ans1 = createPivots(np.array(matrix))
ans2 = createPivots(np.array(createMatrix(new_matrix, n ,m)))

if (n > m + 1):
    print("DEPENDENT")
    sys.exit()

if (n == m + 1):
    if (ans2 == 1):
        print("AFFINELY INDEPENDENT")
    else:
        print("DEPENDENT")
    sys.exit()

if (n <= m):
    if (ans1 == 1):
        print("LINEARLY INDEPENDENT")
    elif (ans2 == 1):
        print("AFFINELY INDEPENDENT")
    else:
        print("DEPENDENT")
    sys.exit()
