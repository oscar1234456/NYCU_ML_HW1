import numpy as np
from matplotlib import pyplot as plt

from config import constant
from data_process import dataset
from method import lse, newthon
from print_process import pretty_output, pretty_plot

file_name = input("The file name:")
while True:
    n = int(input("n:"))
    lamb = int(input("lambda:"))
    test_data = dataset.Dataset(constant.DATA_FILE_PATH, file_name)
    test_datapoint_x = test_data.get_data_x()
    A = test_data.get_design_matrix(n)
    test_datapoint_b = test_data.get_data_y()


    lse_w, lse_loss = lse(A, test_datapoint_b, lamb)
    newthon_w, newthon_loss = newthon(A, test_datapoint_b)
    pretty_output(lse_w, lse_loss, method="lse")
    pretty_output(newthon_w, newthon_loss, method="newthon")
    pretty_plot(test_datapoint_x, test_datapoint_b, lse_w, newthon_w)