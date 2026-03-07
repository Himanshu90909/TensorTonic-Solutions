import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Ensure it's treated as a NumPy array
    A = np.array(A)
    
    N, M = A.shape
    
    # Allocate output with correct shape
    result = np.zeros((M, N), dtype=A.dtype)
    
    # Manual transpose
    for i in range(N):
        for j in range(M):
            result[j, i] = A[i, j]
    
    return result
