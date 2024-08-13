import math
import numpy as np

def input_matrix():
    elements = input().split()
    a, b = map(float, elements)
    elements = input().split()
    c, d = map(float, elements)
    return a, b, c, d

def calculate_eigenValues(a, b, c, d):
    determinant = a * d - b * c
    trace = a + d
    delta = math.sqrt(trace ** 2 - 4 * determinant)
    value1 = 0.5 * (trace + delta)
    value2 = 0.5 * (trace - delta)
    return value1, value2

def calculate_ATA(a, b, c, d):
    aPrime = a**2 + c**2
    bPrime = a*b + c*d
    cPrime = a*b + c*d
    dPrime = b**2 + d**2
    return aPrime, bPrime, cPrime, dPrime

def printMatrix(matrix):
    temp = np.array(matrix)
    result = np.round(matrix, decimals=10)

    for i in range (2):
        for j in range(2):
            if (abs(result[i][j]) < 1e-13):
                result[i][j] = 0
    

    print(f"{format(result[0][0],'.2f')} {format(result[0][1],'.2f')}")
    print(f"{format(result[1][0],'.2f')} {format(result[1][1],'.2f')}")

def multiplyMatrix(matrix1, matrix2):
    result = [[0, 0], [0, 0]]

    for i in range(2):
        for j in range(2):
            for k in range(2):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result

def multiplyMatrixVector(matrix, vector):
    result = [0, 0]

    for i in range(2):
        for j in range(2):
            result[i] += matrix[i][j] * vector[j]

    return result

def normalize(vector):
    x, y = vector[0], vector[1]
    sum = math.sqrt(x**2 + y**2)
    normalizedVector = [x/sum, y/sum]
    return normalizedVector

if __name__ == "__main__":
    a, b, c, d = input_matrix()

    A = [[a,b],[c,d]]

    trace = a + d
    value1, value2 = calculate_eigenValues(a, b, c, d)

    safety = "safe"
    if (abs(value1) > 2 or abs(value2) > 2):
        safety = "unsafe"

    aP, bP, cP, dP = calculate_ATA(a, b, c, d)
    valueP1, valueP2, = calculate_eigenValues(aP, bP, cP, dP)

    maxValue = max(valueP1,valueP2)
    minValue = min(valueP1,valueP2)

    sigma = [[math.sqrt(maxValue), 0.00], [0.00, math.sqrt(minValue)]]

   
    if (value1 == value2):
        eigenvector1 = [1,0]
        eigenvector2 = [0,1]
    else:
        eigenvector1 = normalize([1, (valueP1 - aP) / bP])
        eigenvector2 = normalize([1, (valueP2 - aP) / bP])

    vT = [eigenvector1, eigenvector2]

    mEV1 = multiplyMatrixVector(A,eigenvector1)
    mEV2 = multiplyMatrixVector(A,eigenvector2)


    sV1 = math.sqrt(valueP1)
    sV2 = math.sqrt(valueP2)

    if (sV1 < 1e-10):
        sV1 = 1e10
    if (sV2 < 1e-10):
        sV2 = 1e10

    U = [[mEV1[0] / sV1, mEV2[0] / sV2], [mEV1[1] / sV1, mEV2[1] / sV2]]
    product = multiplyMatrix(multiplyMatrix(U, sigma), vT)

    print(format(round(trace,2),'.2f'))
    print(safety)
    printMatrix(sigma)
    printMatrix(product)