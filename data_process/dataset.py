import os.path
import numpy as np


class Dataset:
    def __init__(self, file_path: str, file_name: str):
        self.file_path = file_path
        self.file_name = file_name
        self.x = None
        self.y = None
        self.data_size: int = 0
        self._get_data()  # read values from the file

    def _get_data(self):
        print("=====Getting Data=====")
        file_path_name = self.file_path + self.file_name
        temp_x = list()
        temp_y = list()
        assert os.path.isfile(file_path_name), "The file path is wrong!"  # have error if the path is wrong
        with open(file_path_name) as f:  # read data (with the format that was announced in HW notice)
            for line in f.readlines():
                s = line.strip("\n")
                s = s.split(",")
                temp_x.append(float(s[0]))
                temp_y.append(float(s[1]))
        self.x = np.array(temp_x, dtype=float)  # transfer to numpy array (with float type)
        self.y = np.array(temp_y, dtype=float)
        self.data_size = len(temp_x)
        print(f"==>#Datapoints: {self.data_size}")

    def get_data_x(self):
        return self.x

    def get_data_y(self):
        return self.y

    def get_design_matrix(self, n:int =1):
        # made m=#data by n design matrix
        # | 1 x1 x1^2...|
        # | 1 x2 x2^2...|
        # | 1 x3 x3^2...|
        A = np.ones((self.data_size, n))
        for exp in range(1, n):
            A[ :,exp] = A[ :,exp-1] * self.x
        return A


if __name__ == "__main__":
    d = Dataset("../data/", "test1.txt")
    a = d.get_design_matrix(3)
    print(a)
