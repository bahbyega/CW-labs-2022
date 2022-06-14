from scipy import linalg as la
import math

def spectral_condition_nums(matr):
    return la.norm(matr) * la.norm(la.inv(matr))

def bulk_condition_nums(matr):
    product = 1
    n, m = matr.shape
    
    for i in range(n):
        temp_sum = 0
        for j in range(m):
            temp_sum += pow(matr[i, j], 2)
        product *= math.sqrt(temp_sum)
    
    return product / abs(la.det(matr))

def angular_condition_nums(matr):
    iterator = zip(matr, la.inv(matr).transpose())
    return max([la.norm(row) * la.norm(col) for (row, col) in iterator])

def compute_matr_condition_nums(matr):
    return (
        spectral_condition_nums(matr),
        bulk_condition_nums(matr),
        angular_condition_nums(matr)
    )