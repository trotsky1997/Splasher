def f(x):
    if x == 0:
        return 0
    else:
        return 0.9*f(x - 1) + 1


print(f(100))