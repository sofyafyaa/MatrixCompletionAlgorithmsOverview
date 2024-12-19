import numpy as np


def calculate_relative_error(X, M_true):
    return np.linalg.norm(X - M_true) / np.linalg.norm(M_true)

def calculate_relative_residual(X, M_true, Omega):
    return np.linalg.norm(Omega * X - Omega * M_true) / np.linalg.norm(Omega * M_true)
