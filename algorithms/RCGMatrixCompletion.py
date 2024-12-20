# import numpy as np
import numpy as np
import numpy
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from MatrixCompletionClass import MatrixCompletion
from utils.metrics import calculate_relative_error, calculate_relative_residual


class RCGMatrixCompletion(MatrixCompletion):
    """
    Initialize the RCG Matrix Completion class.

    Parameters:
    - Omega: Binary mask matrix (1 for observed entries, 0 for missing).
    - alpha: Regularization parameter.
    """

    def __init__(self, params_str):
        super().__init__(params_str)
        params = json.loads(params_str)
        self.alpha = params["alpha"]

    def complete_matrix(self, M, Omega, **kwargs):
        """Solve the matrix completion problem using Riemannian Conjugate Gradient."""
        self.iters_info = []

        M = np.asarray(M)
        Omega = np.asarray(Omega)

        method = kwargs["method"]
        metric = kwargs["metric"]
        self.with_laplace = kwargs["laplace"]

        # Initialization
        G, H = self._get_initial_approximation(M, Omega)

        grad_G_prev, grad_H_prev = None, None
        direction_G, direction_H = None, None

        if self.with_laplace:
            theta_r, theta_c = self.compute_laplacians(M * Omega)

        for iter in tqdm(range(self.num_iters)):
            # Compute gradient and cost
            if self.with_laplace:
                grad_G, grad_H = self._compute_gradient(
                    G, H, M, Omega, theta_c, theta_r, metric
                )
            else:
                grad_G, grad_H = self._compute_gradient(G, H, M, Omega, metric)
            cost = self._compute_cost(G, H, M, Omega)

            # Check stopping criterion based on gradient norm
            grad_norm = np.sqrt(
                np.linalg.norm(grad_G) ** 2 + np.linalg.norm(grad_H) ** 2
            )
            if grad_norm < self.tol:
                print(f"Converged at iteration {iter} with cost {cost}")
                break

            # Compute conjugate gradient direction
            if iter == 0 or method == "rgd":
                direction_G, direction_H = -grad_G, -grad_H
            else:
                beta_FR = (np.sum(grad_G**2) + np.sum(grad_H**2)) / (
                    np.sum(grad_G_prev**2) + np.sum(grad_H_prev**2)
                )
                beta_FR = max(beta_FR, 0)  # Ensure non-negative beta

                direction_G = -grad_G + beta_FR * direction_G
                direction_H = -grad_H + beta_FR * direction_H

            # Perform line search to find step size
            step_size = self._line_search(
                G, H, grad_G, grad_H, direction_G, direction_H, M, Omega
            )

            # Update variables using conjugate gradient descent step
            G += step_size * direction_G
            H += step_size * direction_H

            # Store previous gradients for next iteration
            grad_G_prev, grad_H_prev = grad_G.copy(), grad_H.copy()

            X = G @ H.T

            relative_error = calculate_relative_error(X, M)
            relative_residual = calculate_relative_residual(X, M, Omega)
            self.iters_info.append(
                {
                    "iteration": iter,
                    "cost": cost,
                    "grad_norm": grad_norm,
                    "relative_error": relative_error,
                    "relative_residual": relative_residual,
                }
            )

        return G @ H.T

    def compute_laplacians(self, M):
        """
        Compute the row-wise and column-wise Laplacians for a non-square matrix.

        Parameters:
        - M: Input matrix of size (m, n).

        Returns:
        - L_rows: Row-wise Laplacian (m x m).
        - L_columns: Column-wise Laplacian (n x n).
        """
        # Compute row adjacency matrix (MM^T)
        A_rows = np.dot(M, M.T)

        # Degree matrix for rows
        # D_rows = np.eye(A_rows.shape[0])
        D_rows = np.diag(np.sum(A_rows, axis=1))

        # Row-wise Laplacian
        L_rows = D_rows - A_rows

        # Compute column adjacency matrix (M^TM)
        A_columns = np.dot(M.T, M)

        # Degree matrix for columns
        D_columns = np.diag(np.sum(A_columns, axis=1))
        # D_rows = np.diag(np.sum(A_rows, axis=1))

        # Column-wise Laplacian
        L_columns = D_columns - A_columns

        return (
            np.eye(L_rows.shape[0]) + 0.01 * L_rows,
            np.eye(L_columns.shape[0]) + 0.01 * L_columns,
        )

    def _get_initial_approximation(self, M, Omega):
        """Spectral initialization using SVD."""
        U, S, Vt = np.linalg.svd(Omega * M, full_matrices=False)
        G_init = U[:, : self.rank] @ np.diag(np.sqrt(S[: self.rank]))
        H_init = Vt[: self.rank, :].T @ np.diag(np.sqrt(S[: self.rank]))
        return G_init, H_init

    def _compute_gradient(
        self, G, H, M, Omega, laplac_c=None, laplac_r=None, metric="QPRECON"
    ):
        """Compute the gradient of the objective function."""
        residual = Omega * (G @ H.T - M)
        if self.with_laplace:
            grad_G = residual @ H + self.alpha * laplac_r @ G
            grad_H = residual.T @ G + self.alpha * laplac_c @ H
        else:
            grad_G = residual @ H + self.alpha * G
            grad_H = residual.T @ G + self.alpha * H
        if metric == "QPRECON":
            rgrad_G, rgrad_H = self.compute_qprecon_gradient(G, H, grad_G, grad_H)
        else:
            rgrad_G, rgrad_H = self.compute_qrightinv_gradient(G, H, grad_G, grad_H)

        return rgrad_G, rgrad_H

    def compute_qrightinv_gradient(self, G, H, grad_G, grad_H):
        """
        Compute the Riemannian gradient under QRIGHT-INV metric.

        Parameters:
        - G: Matrix G
        - H: Matrix H
        - grad_G: Euclidean gradient with respect to G
        - grad_H: Euclidean gradient with respect to H

        Returns:
        - rgrad_G: Riemannian gradient with respect to G
        - rgrad_H: Riemannian gradient with respect to H
        """
        GTG = G.T @ G + self.alpha * np.eye(G.shape[1])
        HTH = H.T @ H + self.alpha * np.eye(H.shape[1])

        rgrad_G = grad_G @ GTG
        rgrad_H = grad_H @ HTH

        return rgrad_G, rgrad_H

    def compute_qprecon_gradient(self, G, H, grad_G, grad_H):
        """
        Compute the Riemannian gradient under QPRECON metric.

        Parameters:
        - G: Matrix G
        - H: Matrix H
        - grad_G: Euclidean gradient with respect to G
        - grad_H: Euclidean gradient with respect to H

        Returns:
        - rgrad_G: Riemannian gradient with respect to G
        - rgrad_H: Riemannian gradient with respect to H
        """
        GTG_inv = np.linalg.inv(G.T @ G + self.alpha * np.eye(G.shape[1]))
        HTH_inv = np.linalg.inv(H.T @ H + self.alpha * np.eye(H.shape[1]))

        rgrad_G = grad_G @ HTH_inv
        rgrad_H = grad_H @ GTG_inv

        return rgrad_G, rgrad_H

    def _compute_cost(self, G, H, M, Omega):
        """Compute the cost function."""
        residual = Omega * (G @ H.T - M)
        frob_norm_squared = np.sum(residual**2)
        reg_term = self.alpha * (np.linalg.norm(G) ** 2 + np.linalg.norm(H) ** 2)
        return 0.5 * frob_norm_squared + 0.5 * reg_term

    def _line_search(self, G, H, grad_G, grad_H, direction_G, direction_H, M, Omega):
        """Perform exact line search to find optimal step size."""
        # Compute coefficients for the quartic equation
        A1 = np.sum((Omega * (direction_G @ direction_H.T)) ** 2)
        A2 = np.sum(
            (Omega * (G @ direction_H.T + direction_G @ H.T))
            * (Omega * (direction_G @ direction_H.T))
        )

        B1 = np.sum((Omega * (G @ direction_H.T + direction_G @ H.T)) ** 2)

        C1 = np.sum((Omega * (G @ H.T - M)) * (Omega * (direction_G @ direction_H.T)))

        # Solve for step size s using a cubic/quartic minimization approach
        numerator = -C1
        denominator = A1 + A2 + B1

        if denominator == 0:
            return 1e-4  # Fallback small step size

        step_size = numerator / denominator
        return max(step_size, 1e-4)  # Ensure positive step size

    @staticmethod
    def plot_info(path, experiments):
        # Create and save the plots in a single figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        colors = ["red", "blue", "yellow", "green"]

        axs[0, 0].set_yscale("log")
        axs[0, 0].set_title("Reconstruction Error Over Time", fontsize=18)
        axs[0, 0].set_xlabel("Iteration", fontsize=12)
        axs[0, 0].set_ylabel(r"$\frac{\|X-A\|_F}{\|A\|_F}$", fontsize=16)

        axs[0, 1].set_yscale("log")
        axs[0, 1].set_title("Reconstruction Residual Over Time", fontsize=18)
        axs[0, 1].set_xlabel("Iteration", fontsize=12)
        axs[0, 1].set_ylabel(
            r"$\frac{\|P_{\Omega}(X-A)\|_F}{\|P_{\Omega}(A)\|_F}$", fontsize=16
        )

        axs[1, 0].set_yscale("log")
        axs[1, 0].set_title("Gradient Norm over time", fontsize=18)
        axs[1, 0].set_xlabel("Iteration", fontsize=12)
        axs[1, 0].set_ylabel(r"$\|\nabla\|P_{\Omega}(X-A)\|_F\|$", fontsize=16)

        axs[1, 1].set_yscale("log")
        axs[1, 1].set_title("Conjugate Direction Norm over time", fontsize=18)
        axs[1, 1].set_xlabel("Iteration", fontsize=12)
        axs[1, 1].set_ylabel(r"$\|\eta\|_F$", fontsize=16)

        for i, experiment in enumerate(experiments):
            iters_info = experiment["iters_info"]
            alpha = experiment["alpha"]
            OS = experiment["OS"]
            rank = experiment["rank"]

            iterations = [info["iteration"] for info in iters_info]
            costs = [info["cost"] for info in iters_info]
            grad_norms = [info["grad_norm"] for info in iters_info]
            relative_errors = [info["relative_error"] for info in iters_info]
            relative_residuals = [info["relative_residual"] for info in iters_info]

            label = f"alpha={alpha} missing={OS} rank={rank}"
            color = colors[i % len(colors)]
            axs[0, 0].plot(iterations, relative_errors, label=label, color=color)
            axs[0, 1].plot(iterations, relative_residuals, label=label, color=color)
            axs[1, 0].plot(iterations, grad_norms, label=label, color=color)
            axs[1, 1].plot(iterations, costs, label=label, color=color)

        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[1, 0].legend()
        axs[1, 1].legend()

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        plt.savefig(path)
