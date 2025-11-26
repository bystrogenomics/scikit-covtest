import numpy as np


def a_1_hat(Y):
    S = np.cov(Y.T)
    a_1 = np.trace(S) / S.shape[0]
    return a_1


def a_2_hat(Y):
    n, p = Y.shape
    f = p * n * (n - 1) * (n - 2) * (n - 3)
    return f
