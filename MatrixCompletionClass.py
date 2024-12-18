import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import time
import json


class MatrixCompletion:
    def __init__(self, params_str):
        """
        params_str:
        {
            "m": 100, - rows of M
            "n": 100, - columns of M
            "rank": 10, - rank of M
            "missing_fraction": 0.5, - fraction od deleting values
            "noise_level": 0.1, - sigma of added Gaussian noise
            "num_iters": 100, - Maximum number of iterations
            "tol": 1e-4, - Convergence criteria
            "random_state": 42,
        }
        """
        params = json.loads(params_str)
        self.params_json = params
        self.m = params.get("m", 100)
        self.n = params.get("n", 100)
        self.rank = params.get("rank", 10)
        self.missing_fraction = params.get("missing_fraction", 0.5)
        self.noise_level = params.get("noise_level", 0.1)
        self.num_iters = params.get("num_iters", 100)
        self.tol = params.get("tol", -1)
        self.random_state = params.get("random_state", None)
        # self._set_seed()

        self.error_history = []
        self.time_history = []

    def complete_matrix(self):
        raise NotImplementedError("Implement 'complete_matrix' function.")

    def plot_matrices(self):

        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        plt.title("Original Matrix")
        plt.imshow(self.M_true, aspect="auto", cmap="viridis")
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title("Matrix with Missing Entries and Noise")
        M_display = self.M_noisy.copy()
        M_display[np.isnan(self.M_missing)] = np.nanmean(self.M_missing)
        plt.imshow(M_display, aspect="auto", cmap="viridis")
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title("Completed Matrix")
        plt.imshow(self.M_completed, aspect="auto", cmap="viridis")
        plt.colorbar()

        plt.tight_layout()
        plt.show()

    def plot_error_iteration(self):
        plt.figure(figsize=(18, 5))
        plt.plot(self.error_history, marker="o")
        plt.title("Reconstruction Error Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Frobenius Norm of Reconstruction Error")
        plt.yscale("log")
        plt.grid(True)
        plt.show()

    def plot_error_time(self):
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 2, 2)
        plt.plot(self.time_history, self.error_history, marker="o")
        plt.title("Reconstruction Error Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Frobenius Norm of Reconstruction Error")
        plt.yscale("log")
        plt.grid(True)
        plt.show()

    def run(self):
        """
        Executes pipeline.
        """
        self.create_low_rank_matrix()
        self.remove_entries()
        self.add_gaussian_noise()
        self.complete_matrix()


class SimpleLS(MatrixCompletion):
    def complete_matrix(self):
        """
        Это написало гпт
        Different methods of matrix completion
        In this example LS implemented
        """

        raise NotImplementedError("Implement 'complete_matrix' function.")

        m, n = self.M_noisy.shape
        U = np.random.randn(m, self.rank)
        V = np.random.randn(n, self.rank)
        M_filled = np.nan_to_num(self.M_noisy)
        self.history = []
        self.time = []

        start_time = time.time()
        for it in range(self.num_iters):
            # Update U
            for i in range(m):
                V_i = V[self.M_mask[i, :], :]  # Observed entries in row i
                M_i = M_filled[i, self.M_mask[i, :]]
                if V_i.size == 0:
                    continue
                # Solve least squares: minimize ||U[i]V_i^T - M_i||^2 + lambda_reg ||U[i]||^2
                U[i, :] = np.linalg.lstsq(V_i, M_i, rcond=None)[0]

            # Update V
            for j in range(n):
                U_j = U[self.M_mask[:, j], :]  # Observed entries in column j
                M_j = M_filled[self.M_mask[:, j], j]
                if U_j.size == 0:
                    continue
                # Solve least squares: minimize ||U_j V[j] - M_j||^2 + lambda_reg ||V[j]||^2
                V[j, :] = np.linalg.lstsq(U_j, M_j, rcond=None)[0]

            # Compute current approximation
            M_current = U @ V.T

            # Calculate error
            error = np.linalg.norm((M_current - M_filled)[self.M_mask])
            self.error_history.append(error)
            self.time_history.append(time.time() - start_time)

        self.M_completed = U @ V.T
