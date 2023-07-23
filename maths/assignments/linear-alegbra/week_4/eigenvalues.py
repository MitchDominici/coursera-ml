from numpy.linalg import linalg
import numpy as np
from sympy import *

rows = [[2, 3],[2, 1]]

matrix = np.array(rows, dtype=np.dtype(float))



A_reflection_yaxis = np.flipud(matrix)

w, v = np.linalg.eig(A_reflection_yaxis)


print(w)