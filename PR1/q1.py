import numpy as np
import sys

def input_matrix(n):
    matrix = []
    for i in range(n):
        row = input()
        row = list(map(int, row.split()))
        matrix.append(row)

    identity_matrix = np.identity(n)
    extended_matrix = np.hstack((matrix, identity_matrix))

    return extended_matrix

def createPivot(index, n, matrix):
    flag = False
    if matrix[index][index] == 0:
        for i in range (index + 1, n):
            
            if matrix[i][i] != 0:
                matrix[[index, i]] = matrix[[i, index]]
                
                flag = True
                break
    else:
        flag = True
    if flag == False:
        return 0
    for i in range (index + 1, n):
        mult = matrix[i][index] / matrix[index][index]
        matrix[i] -= mult * matrix[index]
    return 1
            
    

def createPivots(n, matrix):
    for i in range(n):
        answer = createPivot(i, n, matrix)
        if (answer == 0):
            print("no unique solution")
            sys.exit()
        

    for i in range(n - 1, 0, -1):
        j = 0
        while (j != i):
            mult = matrix[j][i] / matrix[i][i]
            matrix[j] -= mult * matrix[i]
            j = j + 1

    for i in range (n):
        matrix[i] = matrix[i] / matrix[i][i]

    return matrix

        
            

# Input for the value of 'n'
n = int(input())

input = input_matrix(n)

matrix = createPivots(n, np.array(input))


    
np.set_printoptions(suppress=True)
rounded_matrix = np.empty_like(matrix)
for i in range (n):
    for j in range (2*n + 1):
        rounded_matrix[i][j] = round(matrix[i][j],2)
print(np.round(np.sort(matrix[:, n])).astype(int))
print(np.array(rounded_matrix[:, -n:]))

