import matplotlib.pyplot as plt
import numpy as np


def pretty_plot(test_datapoint_x: np.ndarray, test_datapoint_b, lse_w, newthon_w):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Fitting Curve')
    n = lse_w.shape[0]
    test_datapoint_x_min = test_datapoint_x.min()
    test_datapoint_x_max = test_datapoint_x.max()
    x_space = np.linspace(test_datapoint_x_min-1, test_datapoint_x_max+1)
    y = np.zeros(len(x_space))
    ax1.scatter(test_datapoint_x, test_datapoint_b, c = "red")
    ax2.scatter(test_datapoint_x, test_datapoint_b, c="red")
    for i in range(n):
        y = y + lse_w[i] * np.power(x_space, i)
    ax1.plot(x_space, y, c="black")
    ax1.set_title("LSE")
    y = np.zeros(len(x_space))
    for i in range(n):
        y = y + newthon_w[i] * np.power(x_space, i)
    ax2.plot(x_space, y, c="black")
    ax2.set_title("Newthon's method")
    fig.tight_layout()

    plt.show()