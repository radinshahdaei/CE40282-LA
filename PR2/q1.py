import numpy as np

def gram_schmidt(matrix):
    # Get the number of rows and columns in the matrix
    rows, cols = matrix.shape
    
    # Create an empty matrix to store the orthonormal basis
    orthonormal_basis = np.zeros((rows, cols))
    
    for i in range(cols):
        # Copy the i-th column to the orthonormal basis
        orthonormal_basis[:, i] = matrix[:, i]
        
        # Subtract the projections of the previous vectors
        for j in range(i):
            orthonormal_basis[:, i] -= np.dot(matrix[:, i], orthonormal_basis[:, j]) / np.dot(orthonormal_basis[:, j], orthonormal_basis[:, j]) * orthonormal_basis[:, j]
        
        # Normalize the new vector
        orthonormal_basis[:, i] /= np.linalg.norm(orthonormal_basis[:, i])
    
    return orthonormal_basis

# Input the number of vectors and dimension

input_str = input()
num_vectors, dimension = map(int, input_str.split())

# Input the vectors
vectors = []
for i in range(num_vectors):
    vector = [float(x) for x in input().split(' ')]
    vectors.append(vector)

# Convert the input to a NumPy matrix
matrix = np.array(vectors).T

# Perform Gram-Schmidt process
orthonormalized_matrix = gram_schmidt(matrix)

new_matrix = np.transpose(orthonormalized_matrix)

sorted_matrix = sorted(new_matrix, key=lambda x: (-x[0], -x[1], -x))

np.set_printoptions(precision=3, suppress=True)

# Display results


for row in sorted_matrix:
    for element in row:
        print("{:.3f}".format(element), end=" ")
    print()