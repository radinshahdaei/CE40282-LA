def calculate_inverse(matrix):
    # Check if the matrix is square
    if len(matrix) != len(matrix[0]):
        print("Error: The matrix is not square. Inverse does not exist.")
        return None

    n = len(matrix)
    
    # Create an identity matrix of the same size as the input matrix
    identity_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        identity_matrix[i][i] = 1

    # Perform Gaussian elimination with partial pivoting
    for i in range(n):
        # Find the pivot row
        pivot_row = max(range(i, n), key=lambda r: abs(matrix[r][i]))

        # Swap rows
        matrix[i], matrix[pivot_row] = matrix[pivot_row], matrix[i]
        identity_matrix[i], identity_matrix[pivot_row] = identity_matrix[pivot_row], identity_matrix[i]

        # Scale the pivot row to have a leading 1
        pivot_element = matrix[i][i]
        if pivot_element != 0:
            matrix[i] = [elem / pivot_element for elem in matrix[i]]
            identity_matrix[i] = [elem / pivot_element for elem in identity_matrix[i]]

        # Eliminate other rows
        for j in range(n):
            if j != i:
                factor = matrix[j][i]
                matrix[j] = [elem_j - factor * elem_i for elem_i, elem_j in zip(matrix[i], matrix[j])]
                identity_matrix[j] = [elem_j - factor * elem_i for elem_i, elem_j in zip(identity_matrix[i], identity_matrix[j])]

    # Check if the original matrix is singular (diagonal has zeros)
    if any(matrix[i][i] == 0 for i in range(n)):
        print("Error: The matrix is singular and does not have an inverse.")
        return None

    return identity_matrix

def transpose(matrix):
    # Calculate the transpose of the matrix
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def matrix_multiply(matrix1, matrix2):
    # Calculate the product of two matrices
    result = [[sum(matrix1[i][k] * matrix2[k][j] for k in range(len(matrix2))) for j in range(len(matrix2[0]))] for i in range(len(matrix1))]
    return result

def calculate_AT_A(matrix):
    # Calculate transpose of the matrix A
    A_transpose = transpose(matrix)
    
    # Calculate the product A^T A
    result = matrix_multiply(A_transpose, matrix)
    
    return result

def matrix_vector_multiply(matrix, vector):
    # Check if the dimensions are compatible for multiplication
    if len(matrix[0]) != len(vector):
        print("Error: Incompatible dimensions for matrix-vector multiplication.")
        return None

    # Perform matrix-vector multiplication
    result = [sum(matrix[i][j] * vector[j] for j in range(len(vector))) for i in range(len(matrix))]
    
    return result

def input_matrix(n ,m):
    try:
        # Initialize an empty matrix
        matrix = []

        # Input the matrix values
        for _ in range(n):
            row = list(map(float, input().split()))
            if len(row) != m:
                raise ValueError("Error: Number of elements in a row does not match the specified 'm'.")
            matrix.append(row)

        return matrix

    except ValueError as e:
        print(str(e))
        return None
    
def multiply_vector_by_scalar(vector, n):
    result_vector = [element * n for element in vector]
    return result_vector

def input_vector():
    try:
        # Input the vector values
        vector = list(map(float, input().split()))
        
        return vector

    except ValueError as e:
        print(str(e))
        return None

if __name__ == "__main__":
    n = int(input())
    m = int(input())
    matrix = input_matrix(n, m)
    y = input_vector()

    ATA = calculate_AT_A(matrix)
    invATA = calculate_inverse(ATA)
    xHat = matrix_vector_multiply(matrix_multiply(invATA, transpose(matrix)),y)
    answer = multiply_vector_by_scalar(xHat, n)

    for element in answer:
        print(format(round(element,2),'.2f'))