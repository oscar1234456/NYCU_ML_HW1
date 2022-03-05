from sklearn.datasets import make_regression

def make_r():
     res = make_regression(1000, 2)
     with open("../data/test4.txt", 'w') as f:
         for i in res[0]:
             f.write(f"{i[0]},")
             f.write(f"{i[1]}\n")
     print("end")

if __name__ == "__main__":
    make_r()
