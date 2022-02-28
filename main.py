from config import constant
from data_process import dataset

file_name = input("The file name:")
n = input("n:")
lam = input("lambda:")
test_data = dataset.Dataset(constant.DATA_FILE_PATH, file_name)

A = test_data.get_design_matrix(int(n))


