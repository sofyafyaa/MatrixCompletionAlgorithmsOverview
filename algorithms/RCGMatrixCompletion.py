import numpy as np
import json

from MatrixCompletionClass import MatrixCompletion


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
        self.alpha = params['alpha']
    
    def complete_matrix(self, M, Omega):
        """Solve the matrix completion problem using Riemannian Conjugate Gradient."""
        # Initialization
        G, H = self._get_initial_approximation(M, Omega)
        
        grad_G_prev, grad_H_prev = None, None
        direction_G, direction_H = None, None
        
        for iter in range(self.num_iters):
            # Compute gradient and cost
            grad_G, grad_H = self._compute_gradient(G, H, M, Omega)
            cost = self._compute_cost(G, H, M, Omega)
            
            # Check stopping criterion based on gradient norm
            grad_norm = np.sqrt(np.linalg.norm(grad_G) ** 2 + np.linalg.norm(grad_H) ** 2)
            if grad_norm < self.tol:
                print(f"Converged at iteration {iter} with cost {cost}")
                break
            
            # Compute conjugate gradient direction
            if iter == 0:
                direction_G, direction_H = -grad_G, -grad_H
            else:
                beta_FR = (
                    np.sum(grad_G**2) + np.sum(grad_H**2)
                ) / (
                    np.sum(grad_G_prev**2) + np.sum(grad_H_prev**2)
                )
                beta_FR = max(beta_FR, 0)  # Ensure non-negative beta
                
                direction_G = -grad_G + beta_FR * direction_G
                direction_H = -grad_H + beta_FR * direction_H
            
            # Perform line search to find step size
            step_size = self._line_search(
                G,
                H,
                grad_G,
                grad_H,
                direction_G,
                direction_H,
                M,
                Omega
            )
            
            # Update variables using conjugate gradient descent step
            G += step_size * direction_G
            H += step_size * direction_H
            
            # Store previous gradients for next iteration
            grad_G_prev, grad_H_prev = grad_G.copy(), grad_H.copy()
            
            # Print progress every few iterations
            if iter % 10 == 0:
                print(f"Iter {iter}: Cost={cost:.6f}, GradNorm={grad_norm:.6e}")
        
        return G @ H.T
    
    def _get_initial_approximation(self, M, Omega):
        """Spectral initialization using SVD."""
        U, S, Vt = np.linalg.svd(Omega * M, full_matrices=False)
        G_init = U[:, :self.rank] @ np.diag(np.sqrt(S[:self.rank]))
        H_init = Vt[:self.rank, :].T @ np.diag(np.sqrt(S[:self.rank]))
        return G_init, H_init
    
    def _compute_gradient(self, G, H, M, Omega):
        """Compute the gradient of the objective function."""
        residual = Omega * (G @ H.T - M)
        grad_G = residual @ H + self.alpha * G
        grad_H = residual.T @ G + self.alpha * H
        return grad_G, grad_H

    def _compute_cost(self, G, H, M, Omega):
        """Compute the cost function."""
        residual = Omega * (G @ H.T - M)
        frob_norm_squared = np.sum(residual ** 2)
        reg_term = self.alpha * (np.linalg.norm(G) ** 2 + np.linalg.norm(H) ** 2)
        return 0.5 * frob_norm_squared + 0.5 * reg_term

    def _line_search(self, G, H, grad_G, grad_H, direction_G, direction_H, M, Omega):
        """Perform exact line search to find optimal step size."""
        # Compute coefficients for the quartic equation
        A1 = np.sum((Omega * (direction_G @ direction_H.T)) ** 2)
        A2 = np.sum((Omega * (G @ direction_H.T + direction_G @ H.T)) *
                    (Omega * (direction_G @ direction_H.T)))
        
        B1 = np.sum((Omega * (G @ direction_H.T + direction_G @ H.T)) ** 2)
        
        C1 = np.sum((Omega * (G @ H.T - M)) *
                    (Omega * (direction_G @ direction_H.T)))
        
        # Solve for step size s using a cubic/quartic minimization approach
        numerator = -C1
        denominator = A1 + A2 + B1
        
        if denominator == 0:
            return 1e-4  # Fallback small step size
        
        step_size = numerator / denominator
        return max(step_size, 1e-4)  # Ensure positive step size
