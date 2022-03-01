import matplotlib.pyplot as plt
import numpy as np


def pretty_plot(test_datapoint_x: np.ndarray, test_datapoint_b, w):
    n = w.shape[0]
    test_datapoint_x_min = test_datapoint_x.min()
    test_datapoint_x_max = test_datapoint_x.max()
    x_space = np.linspace(test_datapoint_x_min-1, test_datapoint_x_max+1, 500)
    y = np.zeros(len(x_space))
    plt.scatter(test_datapoint_x, test_datapoint_b, c = "red")
    for i in range(n):
        y = y + w[i] * np.power(x_space, i)
    plt.plot(x_space, y, c="black")
    plt.show()