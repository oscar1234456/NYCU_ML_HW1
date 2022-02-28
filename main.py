from config import constant
from data_process import dataset

file_name = input("The file name:")
test_data = dataset.Dataset(constant.DATA_FILE_PATH, file_name)
a = test_data.get_data_x()
b = test_data.get_data_y()
print(test_data.get_data_x())
print(test_data.get_data_y())

