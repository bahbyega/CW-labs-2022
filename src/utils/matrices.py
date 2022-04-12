from scipy import linalg as la
from scipy import sparse
import numpy as np

def round_matrices(matrices, decimals):
    return [matrix.round(decimals) for matrix in matrices]

def create_hilbert_matrix(size):
    return la.hilbert(size)

def create_hilbert_matrices(range):
    return [create_hilbert_matrix(i) for i in range]

def create_tridiagonal_matrix(size):
    if size > 2:
        zeros = [0 for _ in range(size - 2)]
    else:
        raise AttributeError('size should be greater than 2.')
    return np.array(la.toeplitz([2, -1] + zeros, [0, -1] + zeros))

def create_tridiagonal_matrices(range):
    return [create_tridiagonal_matrix(i) for i in range]

def create_random_matrix(size):
    return np.random.random((size, size))

def create_random_matrices(range):
    return [create_random_matrix(i) for i in range]

def create_random_sparse_matrix(size):
    return np.matrix(sparse.random(size, size, density=0.3).toarray())

def create_random_sparse_matrices(range):
    return [create_random_sparse_matrix(i) for i in range]