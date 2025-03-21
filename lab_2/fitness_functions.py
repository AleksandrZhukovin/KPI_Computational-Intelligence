import numpy as np


__all__ = (
    'ackley',
    'rosenbrock',
    'cross_in_tray',
    'holder_table',
    'mccormick',
    'styblinski_tang',
)


def ackley(X):
    a = 20
    b = 0.2
    c = 2 * np.pi
    x1, x2 = X
    return -a * np.exp(-b * np.sqrt(0.5 * (x1**2 + x2**2))) - np.exp(0.5 * (np.cos(c*x1) + np.cos(c*x2))) + a + np.e


def rosenbrock(X, a=1, b=100):
    x1, x2 = X
    return (a - x1)**2 + b * (x2 - x1**2)**2


def cross_in_tray(X):
    x1, x2 = X
    term1 = np.sin(x1) * np.sin(x2)
    term2 = np.exp(abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))
    return -0.0001 * (abs(term1 * term2) + 1) ** 0.1


def holder_table(X):
    x1, x2 = X
    term1 = np.sin(x1) * np.cos(x2)
    term2 = np.exp(abs(1 - np.sqrt(x1**2 + x2**2) / np.pi))
    return -abs(term1 * term2)


def mccormick(X):
    x1, x2 = X
    return np.sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1


def styblinski_tang(X):
    x1, x2 = X
    return 0.5 * ((x1**4 - 16*x1**2 + 5*x1) + (x2**4 - 16*x2**2 + 5*x2))
