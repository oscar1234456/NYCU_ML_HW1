import numpy as np

from linear_al.inverse import inverse


def lse(A, b, lamb):
    # maybe A is not a symmetric matrix
    # but, A.T@A should be a symmetric square matrix
    # Moreover, A.T@A + lambdaI is a positive matrix, invertible
    A_row_len, A_col_len = A.shape
    lambI = lamb * np.identity(A_col_len)  # notice: A.T@A is (A_col_len, A_col_len)
    # w = inverse(A.T@A+lambI)@A.T@b (closed form)
    w = inverse(A.T @ A + lambI) @ A.T @ b

    final = A @ w
    loss = np.sum(np.power(final - b, 2))  # cal loss (Squared Error), b: Ground Truth

    return w, loss
