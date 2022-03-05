from config import constant
from data_process import dataset
from method import lse, newton
from print_process import pretty_output, pretty_plot

file_name = input("The file name:")  # input the text data name

while True:
    n = int(input("n:"))  # input the degree of the polynomial
    lamb = int(input("lambda:"))  # input the lambda (the weight of regularization)
    test_data = dataset.Dataset(constant.DATA_FILE_PATH, file_name)  # create a dataset

    test_datapoint_x = test_data.get_data_x()
    A = test_data.get_design_matrix(n)  # get new design matrix with n degree of the polynomial
    test_datapoint_b = test_data.get_data_y()  # corresponding to the sign in lecture

    lse_w, lse_loss = lse(A, test_datapoint_b, lamb)  # run the least square error method
    newton_w, newton_loss = newton(A, test_datapoint_b)  # run Newton-Raphson method

    pretty_output(lse_w, lse_loss, method="lse")  # show the information with template for lse
    pretty_output(newton_w, newton_loss, method="newton") # show the information with template for Newton-Raphson
    pretty_plot(test_datapoint_x, test_datapoint_b, lse_w, newton_w)  # show the plots

    print("---next run---")
