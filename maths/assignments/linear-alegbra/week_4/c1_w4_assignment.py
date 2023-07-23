import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided


def T_reflection_yaxis(v):
    A = np.array([[-1, 0], [0, 1]])
    w = A @ v

    return w


def transform_vectors(T, v1, v2, v3, v4):
    V = np.hstack((v1.reshape(4, 1), v2.reshape(4, 1), v3.reshape(4, 1), v4.reshape(4, 1)))
    W = T(V)

    return W


def T_hscaling(v):
    A = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    w = A @ v

    return w


rows = [[2, 3], [2, 1]]

matrix = np.array(rows, dtype=np.dtype(float))

# print(transformation_result_reflection_yaxis)

# for i in range(matrix.shape[1]):
#     matrix[:, i] = np.roll(matrix[:, i], i)

M_shear_x = np.array([[1, 0.5], [0, 1]])
# print("M_shear_x by M_rotation_90_clockwise:\n", transformation_result_reflection_yaxis @ M_shear_x)

rows, cols = matrix.shape
M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
sheared = cv2.warpPerspective(matrix, M, (int(cols), int(rows)))

P2  = np.array([
  [0, 0.4, 0.2, 0.6, 0.1],
  [0.2, 0, 0.3, 0.1, 0.3],
  [0.1, 0.3, 0, 0.2, 0.2],
  [0.5, 0.1, 0.3, 0, 0.4],
  [0.2, 0.2, 0.2, 0.1, 0]
])

# print(sheared)
P = np.array([
    [0, 0.75, 0.35, 0.85],
    [0.15, 0, 0.35, 0.05],
    [0.30, 0.20, 0, 0.1],
    [0.55, 0.05, 0.30, 0],
])


# print(" Result of the transformation (matrix form):\n", transformation_result_hscaling)

X0 = np.array([[0], [0], [0], [1], [0]])


# Multiply matrix P and X_0 (matrix multiplication).
X1 = np.matmul(P2, X0)

### END CODE HERE ###
print(sum(P2))


def shear3(a, strength=1, shift_axis=0, increase_axis=1):
    if shift_axis > increase_axis:
        shift_axis -= 1
    res = np.empty_like(a)
    index = np.index_exp[:] * increase_axis
    roll = np.roll
    for i in range(0, a.shape[increase_axis]):
        index_i = index + (i,)
        res[index_i] = roll(a[index_i], -i * strength, shift_axis)
    return res


res = shear3(P)
