import numpy as np

from linear_al.decomposition import PALDU_decomposition


def inverse(A):
    P, L, D, U = PALDU_decomposition(A)
    # TODO: lower matrix inverse (diagonal element all 1)
    inv_L = lower_matrix_inverse(L)
    # TODO: upper matrix inverse (diagonal element all 1)
    inv_U = upper_matrix_inverse(U)
    # TODO: diagonal matrix inverse (the reciprocal of diagonal element)
    inv_D = diagonal_matrix_inverse(D)
    # TODO: target-> inv(A) = inv(U)@inv(D)@inv(L)@P
    # notice: inv(P) = P
    return inv_U@inv_D@inv_L@P


def lower_matrix_inverse(L):
    # L must be a square matrix (all permutation well, and all diagonal elements are 1)
    L_row_len, L_col_len = L.shape  # the lengths of row and col should be the same
    right_matrix = np.identity(L_row_len)
    # Gaussian-Jordan process
    for now_focus_pivot in range(L_col_len - 1):
        for row_cursor in range(now_focus_pivot + 1, L_row_len):
            multiplier = L[row_cursor, now_focus_pivot]
            # pivot always is set to 1
            right_matrix[row_cursor, :] = right_matrix[row_cursor, :] - (multiplier * right_matrix[now_focus_pivot, :])
    return right_matrix


def upper_matrix_inverse(U):
    # U must be a square matrix (all permutation well, and all diagonal elements are 1)
    U_row_len, U_col_len = U.shape # the lengths of row and col should be the same
    right_matrix = np.identity(U_row_len)
    # Gaussian-Jordan process
    for now_focus_pivot in range(U_col_len-1, 0, -1):
        for row_cursor in range(now_focus_pivot-1, -1, -1):
            multiplier = U[row_cursor, now_focus_pivot]
            # pivot always is set to 1
            right_matrix[row_cursor, :] = right_matrix[row_cursor, :] - (multiplier * right_matrix[now_focus_pivot, :])
    return right_matrix


def diagonal_matrix_inverse(D):
    D_row_len, D_col_len = D.shape
    right_matrix = D.copy()
    for row_focus in range(D_row_len):
        right_matrix[row_focus, row_focus] = 1.0 / right_matrix[row_focus, row_focus]
    return right_matrix


if __name__ == "__main__":
    # L = np.array([[1, 0, 0], [2, 1, 0], [8, 2, 1]])
    # rm = lower_matrix_inverse(L)
    # U = np.array([[1, 2, 2], [0, 1, 2], [0, 0, 1]])
    # D = np.array([[2,0,0],[0,1,0],[0,0,3]], dtype = float)
    # rm = diagonal_matrix_inverse(D)
    A = np.array([[2,1,0],[1,2,1],[0,1,2]], dtype = float)
    rm = inverse(A)
    print()
