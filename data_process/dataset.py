import os.path
import numpy as np


class Dataset:
    def __init__(self, file_path: str, file_name: str):
        self.file_path = file_path
        self.file_name = file_name
        self.x = None
        self.y = None
        self._get_data()

    def _get_data(self):
        print("=====Getting Data=====")
        file_path_name = self.file_path + self.file_name
        temp_x = list()
        temp_y = list()
        assert os.path.isfile(file_path_name), "The file path is wrong!"
        with open(file_path_name) as f:
            for line in f.readlines():
                s = line.strip("\n")
                s = s.split(",")
                temp_x.append(float(s[0]))
                temp_y.append(float(s[1]))
        self.x = np.array(temp_x)
        self.y = np.array(temp_y)
        print(f"==>#Datapoints: {len(temp_x)}")

    def get_data_x(self):
        return self.x

    def get_data_y(self):
        return self.y


if __name__ == "__main__":
    pass
    # DataSet("../data/", "testfile.txt")
    # print(os.path.isfile(""))
