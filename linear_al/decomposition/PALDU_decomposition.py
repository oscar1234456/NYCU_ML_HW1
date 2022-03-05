import numpy as np


def PALDU_decomposition(A):
    # A need to be a square matrix
    # Target: PA = LDU
    A_row_len, A_col_len = A.shape

    U = A.copy()  # upper matrix (the outcome of Gaussian process)
    L = np.identity(A_col_len)  # lower matrix (the inverse of elimination matrix)
    P = np.identity(A_col_len)  # permutation matrix
    D = np.identity(A_col_len)  # diagonal matrix

    for now_focus_pivot in range(A_col_len-1):
        max_row_index:int = now_focus_pivot # default: the row index of pivot
        max_row_element:float = abs(U[now_focus_pivot, now_focus_pivot])  # default: the value of pivot
        for row_cursor in range(now_focus_pivot+1, A_row_len):
            if U[row_cursor, now_focus_pivot] > max_row_element:
                # if the value below pivot is larger than pivot(or other larger row)
                max_row_index = row_cursor
                max_row_element = abs(U[row_cursor, now_focus_pivot])

        # swap the matrix by the formula
        # U: change row with pivot and larger one row (once)
        # L: change row and column with pivot and larger one row and column (twice)
        # P: change row or column with pivot and larger one (once)
        U[[max_row_index, now_focus_pivot], now_focus_pivot:] = U[[now_focus_pivot, max_row_index], now_focus_pivot:]
        L[:, [max_row_index, now_focus_pivot]] = L[:, [now_focus_pivot, max_row_index]]  # col change
        L[[max_row_index,now_focus_pivot ], :] = L[[now_focus_pivot, max_row_index], :]  # row change
        P[[max_row_index,now_focus_pivot ], :] = P[[now_focus_pivot, max_row_index], :]

        # Gaussian process
        for row_cursor in range(now_focus_pivot+1, A_row_len):
            multiplier = U[row_cursor, now_focus_pivot]
            pivot = U[now_focus_pivot, now_focus_pivot]
            eliminator = multiplier / pivot

            U[row_cursor, now_focus_pivot:] =  U[row_cursor, now_focus_pivot:] - eliminator * U[now_focus_pivot, now_focus_pivot:]
            L[row_cursor, now_focus_pivot] = eliminator

    # create diagonal matrix (let U diagonal element are all 1)
    for now_focus_pivot in range(A_col_len):
        pivot = U[now_focus_pivot, now_focus_pivot]
        D[now_focus_pivot, now_focus_pivot] = pivot
        U[now_focus_pivot, :] = U[now_focus_pivot, :] / pivot

    return P, L, D, U

if __name__ == "__main__":
    # A = np.array([[2.0,1.0,0.0],[1.0,2.0,1.0],[0.0,1.0,2.0]])
    A = np.array([[2,8],[6,29]])
    # A = np.array([[2,1,0],[1,2,1],[0,1,2]])
    # A = np.array([[2, 4, 5],
    #               [1, 3, 2],
    #               [4, 2, 1]])
    P, L, D, U = PALDU_decomposition(A)
    print()


