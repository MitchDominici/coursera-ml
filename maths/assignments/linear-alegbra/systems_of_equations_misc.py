from numpy.linalg import linalg
import numpy as np
from sympy import *

rows = [[33, -1, -20, ],
        [9, -5, 4, ],
        [-6, 2, 0, ]]

inverst = [[0.5, 0.5, -1, ],
           [0, 0, 1, ],
           [-0.5, 0.5, 1, ]]

# matrix = np.array(rows, dtype=np.dtype(float))
# print(matrix)
# print('\n')


def rank(matrixx):
    rankz = linalg.matrix_rank(matrixx, tol=None, hermitian=False)
    print("rank : {}".format(rankz))


def determinant(matrixx):
    determinantz = np.linalg.det(matrixx)
    print("determinant : {}".format(determinantz))


def echelon_form():
    m = Matrix(rows)
    m_rref = m.rref()
    print("The Row echelon form of matrix M and the pivot columns : {}".format(m_rref))


# AB = [[33, -1, 20], [9, -5, 4], [-6, 2, 0]]
#
# inverse = np.linalg.inv(np.array(AB))
# print("inverse : {}".format(inverse))


matA = [[5, 2, 3],
        [-1, -3, 2],
        [0, 1, -1]]

matB = [[1, 0, -4],
        [2, 1, 0],
        [8, -1, 0]]


def multiply_matrices(A, B):
    matrix_a = np.array(A, dtype=np.dtype(float))
    matrix_b= np.array(B, dtype=np.dtype(float))
    product = np.matmul(matrix_a, matrix_b)
    print("multiply_matrices : {}".format(product))
    return product


mult =  multiply_matrices(matA, matB)

determinant(mult)


vecA = [3, 1, -7]
vecB = [2, 2, 0]
matrixA = np.array(vecA)
matrixB = np.array(vecB)

# print(np.dot(matrixA, matrixB))
