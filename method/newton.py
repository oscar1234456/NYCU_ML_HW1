import numpy as np

from config.constant import TOLERANCE, MAX_Iter
from linear_al.inverse import inverse


def newton(A, b):
    n = A.shape[1]  # get the degree of polynomial
    w_now = np.random.rand(n)
    delta = 99999999  # record the distance between the value of weight with different round
    now_iter = 0
    while delta >= TOLERANCE and now_iter < MAX_Iter:
        print(f"retrieving.... delta:{delta:.5f}")
        gradient_f = 2 * A.T @ A @ w_now - 2 * A.T@b
        inv_H_f = inverse(2 * A.T @ A)  # the differential of gradient_f
        w_next = w_now - (inv_H_f @ gradient_f)  # update the value of weight
        delta = abs(np.sum(w_next - w_now))/n  # count the mean of distance
        w_now = w_next
        now_iter += 1

    final = A @ w_now  # w_now is the weight(or coefficient) of this polynomial
    loss = np.sum(np.power(final - b, 2))  # cal loss (Squared Error), b: Ground Truth

    return w_now, loss
