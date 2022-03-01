import numpy as np

from config.constant import TOLERANCE, MAX_Iter
from linear_al.inverse import inverse


def newthon(A, b):
    # TODO: w (inverse(2*A.T@A)@(2*A.T@A@x0-2*A.T@b))
    n = A.shape[1]
    w_now = np.random.rand(n)
    delta = 99999999
    now_iter = 0
    while delta >= TOLERANCE and now_iter < MAX_Iter:
        print(f"retriveing.... delta:{delta:.5f}")
        gradient_f = 2*A.T@A@w_now-2*A.T@b
        inv_H_f = inverse(2*A.T@A)
        w_next = w_now - (inv_H_f@gradient_f)
        delta = abs(np.sum(w_next - w_now))/n
        w_now = w_next
        now_iter += 1
    final = A @ w_now
    loss = np.sum(np.power(final - b, 2))
    return w_now,loss

    # TODO: iterative
    # TODO: loss
    # TODO: return w, loss