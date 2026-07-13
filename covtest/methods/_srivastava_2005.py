import numpy as np


def a_1_hat(S):
    a_1 = np.trace(S) / S.shape[0]
    return a_1


def a_2_hat(S, n):
    p = S.shape[0]
    frac1 = n**2 / ((n - 1) * (n + 2) * p)
    term1 = np.trace(np.dot(S, S))
    term2 = 1 / n * np.trace(S) ** 2
    a_2 = frac1 * (term1 - term2)
    return a_2


def gamma_1_hat(S, n):
    a_2 = a_2_hat(S, n)
    a_1 = a_1_hat(S)
    gamma_1 = a_2 / a_1**2
    return gamma_1


def gamma_2_hat(S, n):
    a_2 = a_2_hat(S, n)
    a_1 = a_1_hat(S)
    gamma_2 = a_2 - 2 * a_1
    return gamma_2


def T_1_stat(S, n):
    T_1 = gamma_1_hat(S, n) - 1
    return T_1


def T_2_stat(S, n):
    T_2 = gamma_2_hat(S, n) + 1
    return T_2
