import numpy as np
import math
from scipy import linalg as la

def get_sin_cos(a, b):
    if a == 0 and b == 0:
        return 0, 1
    r = math.sqrt(a ** 2 + b ** 2)
    return b / r, -(a / r)

def lu_solve(A, b):
    lu, pivot = la.lu_factor(A)
    sol = la.lu_solve((lu, pivot), b)
    return sol

def u_solve(U, b):
    n, _ = U.shape
    r = b.copy()
    x = np.zeros(n)
    
    for i in reversed(range(n)):
        x[i] = r[i] / U[i, i]
        r[:i] -= U[:i, i] * x[i]
    
    return x