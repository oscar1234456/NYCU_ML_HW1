import numpy as np

from linear_al.decomposition import PALDU_decomposition


def inverse(A):
    P, L, D, U = PALDU_decomposition(A)
    # TODO: lower matrix inverse (diagonal element all 1)
    lower_matrix_inverse(L)
    # TODO: upper matrix inverse (diagonal element all 1)
    upper_matrix_inverse(U)
    # TODO: diagonal matrix inverse (the reciprocal of diagonal element)
    diagonal_matrix_inverse(D)
    # TODO: target-> inv(A) = inv(U)@inv(D)@inv(L)@P
    # inv(P) = P
    pass


def lower_matrix_inverse(L):
    pass


def upper_matrix_inverse(U):
    pass


def diagonal_matrix_inverse(D):
     pass