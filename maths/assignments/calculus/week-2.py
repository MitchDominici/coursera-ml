import numpy as np
# A library for programmatic plot generation.
import matplotlib.pyplot as plt
# A library for data manipulation and analysis.
import pandas as pd
# LinearRegression from sklearn.
from sklearn.linear_model import LinearRegression

path = "C:\\Mitch\\projects\\python-scripts\\data\\tvmarketing.csv"

### START CODE HERE ### (~ 1 line of code)
adv = pd.read_csv(path)
### END CODE HERE ###
X = adv['TV']
Y = adv['Sales']
X_norm = (X - np.mean(X)) / np.std(X)
Y_norm = (Y - np.mean(Y)) / np.std(Y)


def dEdm(m, b, X, Y):
    res = 1 / len(X) * np.dot(m * X + b - Y, X)

    return res


def dEdb(m, b, X, Y):
    res = (1 / len(X)) * np.dot(m * X + b - Y, np.ones(len(X)))
    return res


print(dEdm(0, 0, X_norm, Y_norm))
print(dEdb(0, 0, X_norm, Y_norm))
print(dEdm(1, 5, X_norm, Y_norm))
print(dEdb(1, 5, X_norm, Y_norm))

# Expected Output
#
# -0.7822244248616067
# 5.098005351200641e-16
# 0.21777557513839355
# 5.000000000000002
#
