def pretty_output(w, loss,method: str = "LSE"):
    n = w.shape[0]
    print(f"{method}:")
    print("Fitting line:", end=" ")
    for i in range(n-1, -1, -1):
        if i == 0:
            print(f"{w[i]}")
            break
        else:
            print(f"{w[i]}X^{i}", end="")
        if w[i-1] >= 0:
            print("+", end="")
    print(f"Total error: {loss}")