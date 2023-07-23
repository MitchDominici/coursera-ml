import numpy as np
import w2_unittest

A = np.array([
    [2, -1, 1, 1],
    [1, 2, -1, -1],
    [-1, 2, 2, 2],
    [1, -1, 2, 1]
], dtype=np.dtype(float))
b = np.array([6, 3, 14, 8], dtype=np.dtype(float))
w2_unittest.test_matrix(A, b)

d = np.linalg.det(A)
x = np.linalg.solve(A, b)
w2_unittest.test_det_and_solution_scipy(d, x)

A_test = np.array([
    [1, -2, 3, -4],
    [-5, 6, -7, 8],
    [-4, 3, -2, 1],
    [8, -7, 6, -5]
], dtype=np.dtype(float))


def MultiplyRow(M, row_num, row_num_multiple):
    # .copy() function is required here to keep the original matrix without any changes
    M_new = M.copy()
    # exchange row_num of the matrix M_new with its multiple by row_num_multiple
    # Note: for simplicity, you can drop check if  row_num_multiple has non-zero value, which makes the operation valid
    M_new[row_num] = M_new[row_num] * row_num_multiple
    return M_new


def AddRows(M, row_num_1, row_num_2, row_num_1_multiple):
    M_new = M.copy()
    # multiply row_num_1 by row_num_1_multiple and add it to the row_num_2,
    # exchanging row_num_2 of the matrix M_new with the result
    row_1 = M_new[row_num_1].copy()
    row_2 = M_new[row_num_2].copy()
    M_new[row_num_2] = (row_1 * row_num_1_multiple) + row_2
    return M_new


def SwapRows(M, row_num_1, row_num_2):
    M_new = M.copy()
    # exchange row_num_1 and row_num_2 of the matrix M_new
    row_1 = M_new[row_num_1].copy()
    row_2 = M_new[row_num_2].copy()
    M_new[row_num_1] = row_2
    M_new[row_num_2] = row_1
    return M_new


w2_unittest.test_elementary_operations(MultiplyRow, AddRows, SwapRows)


def augmented_to_ref(A, b):
    ### START CODE HERE ###
    # stack horizontally matrix A and vector b, which needs to be reshaped as a vector (4, 1)
    A_system = np.hstack((A, b.reshape((4, 1))))

    # swap row 0 and row 1 of matrix A_system (remember that indexing in NumPy array starts from 0)
    A_ref = SwapRows(A_system, 0, 1)

    # multiply row 0 of the new matrix A_ref by -2 and add it to the row 1
    A_ref = AddRows(A_ref.copy(), 0, 1, -2)

    # add row 0 of the new matrix A_ref to the row 2, replacing row 2
    A_ref = AddRows(A_ref.copy(), 0, 2, 1)

    # multiply row 0 of the new matrix A_ref by -1 and add it to the row 3
    A_ref = AddRows(A_ref.copy(), 0, 3, -1)

    # add row 2 of the new matrix A_ref to the row 3, replacing row 3
    A_ref = AddRows(A_ref.copy(), 2, 3, 1)

    # swap row 1 and 3 of the new matrix A_ref
    A_ref = SwapRows(A_ref.copy(), 1, 3)

    # add row 2 of the new matrix A_ref to the row 3, replacing row 3
    A_ref = AddRows(A_ref.copy(), 2, 3, 1)

    # multiply row 1 of the new matrix A_ref by -4 and add it to the row 2
    A_ref = AddRows(A_ref.copy(), 1, 2, -4)

    # add row 1 of the new matrix A_ref to the row 3, replacing row 3
    A_ref = AddRows(A_ref.copy(), 1, 3, 1)

    # multiply row 3 of the new matrix A_ref by 2 and add it to the row 2
    A_ref = AddRows(A_ref.copy(), 3, 2, 2)

    # multiply row 2 of the new matrix A_ref by -8 and add it to the row 3
    A_ref = AddRows(A_ref.copy(), 2, 3, -8)

    # multiply row 3 of the new matrix A_ref by -1/17
    A_ref = MultiplyRow(A_ref, 3, -0.05882352941176470588235294117647)

    ### END CODE HERE ###

    return A_ref


A_ref = augmented_to_ref(A, b)

print(A_ref)
w2_unittest.test_augmented_to_ref(augmented_to_ref)

# find the value of x_4 from the last line of the reduced matrix A_ref
x_4 = 1

# find the value of x_3 from the previous row of the matrix. Use value of x_4.
x_3 = A_ref[2, 4] - A_ref[2, 3] * x_4

# find the value of x_2 from the second row of the matrix. Use values of x_3 and x_4
x_2 = A_ref[1, 4] - (A_ref[1, 3] * x_4 + x_3 * A_ref[1, 2])

# find the value of x_1 from the first row of the matrix. Use values of x_2, x_3 and x_4
x_1 = A_ref[0, 4] - (A_ref[0, 3] * x_4 + x_3 * A_ref[0, 2] + x_2 * A_ref[0, 1])
### END CODE HERE ###
print(x_1, x_2, x_3, x_4)
w2_unittest.test_solution_elimination(x_1, x_2, x_3, x_4)


def ref_to_diagonal(A_ref):
    ### START CODE HERE ###
    # multiply row 3 of the matrix A_ref by -3 and add it to the row 2
    A_diag = AddRows(A_ref, 3, 2, -3)

    # multiply row 3 of the new matrix A_diag by -3 and add it to the row 1
    A_diag = AddRows(A_diag, 3, 1, -3)

    # add row 3 of the new matrix A_diag to the row 0, replacing row 0
    A_diag = AddRows(A_diag, 3, 0, 1)

    # multiply row 2 of the new matrix A_diag by -4 and add it to the row 1
    A_diag = AddRows(A_diag, 2, 1, -4)

    # add row 2 of the new matrix A_diag to the row 0, replacing row 0
    A_diag = AddRows(A_diag, 2, 0, 1)

    # multiply row 1 of the new matrix A_diag by -2 and add it to the row 0
    A_diag = AddRows(A_diag, 1, 0, -2)
    ### END CODE HERE ###

    return A_diag


A_diag = ref_to_diagonal(A_ref)

print(A_diag)
w2_unittest.test_ref_to_diagonal(ref_to_diagonal)