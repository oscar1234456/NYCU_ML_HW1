import numpy as np
from matplotlib import pyplot as plt

from config import constant
from data_process import dataset
from method import lse

file_name = input("The file name:")
n = int(input("n:"))
lamb = int(input("lambda:"))
test_data = dataset.Dataset(constant.DATA_FILE_PATH, file_name)
x1 = test_data.get_data_x()
A = test_data.get_design_matrix(n)
b = test_data.get_data_y()

w = lse(A, b, lamb)
print()