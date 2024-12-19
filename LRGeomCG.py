import numpy as np
import matplotlib.pyplot as plt
import json

from collections import defaultdict
from tqdm import tqdm


class TangentVector:
    def __init__(self, U, V, M, Up, Vp, nu=None):
        assert np.allclose(Up.T @ U, 0)
        assert np.allclose(Vp.T @ V, 0)

        self.U = U
        self.V = V

        self.M = M
        self.Up = Up
        self.Vp = Vp

        self.nu = U @ M @ V.T + Up @ V.T + U @ Vp.T if nu is None else nu
        self.norm = np.linalg.norm(self.nu)

    def scalar_product(self, other):
        return np.trace(self.nu.T @ other.nu)

    def __mul__(self, scalar):
        return TangentVector(
            U=self.U, V=self.V,
            M=self.M * scalar,
            Up=self.Up * scalar,
            Vp=self.Vp * scalar,
            nu=self.nu * scalar
        )
    def __truediv__(self, scalar):
        return self * (1.0 / scalar)

    def __neg__(self,):
        return self * -1

    def __add__(self, other):
        assert np.allclose(self.U, other.U)
        assert np.allclose(self.V, other.V)
        return TangentVector(
            U=self.U, V=self.V,
            M=self.M + other.M,
            Up=self.Up + other.Up,
            Vp=self.Vp + other.Vp,
        )

    def __sub__(self, other):
        assert np.allclose(self.U, other.U)
        assert np.allclose(self.V, other.V)
        return self + other * -1

    def __matmul__(self, other):
        assert np.allclose(self.U, other.U)
        assert np.allclose(self.V, other.V)
        pass

class LRGeomCG:
    def __init__(self, params_str):
        """
        params_str:
        {
            "m": 1000, - rows of M
            "n": 1000, - columns of M
            "rank": 10, - rank of M
            "OS": 2, - fraction od deleting values
            "num_iters": 100, - Maximum number of iterations
            "tol": 1e-4, - Convergence criteria
            "random_state": 42,
        }
        """
        params = json.loads(params_str)
        self.params_json = params
        self.m = params.get('m', 1000)
        self.n = params.get('n', 1000)
        self.rank = params.get('rank', 30)
        self.OS = params.get('OS', 2)
        self.num_iters = params.get('num_iters', 100)
        self.tol = params.get('tol', -1)
        self.random_state = params.get('random_state', None)
        self._set_seed()

        self.A = None
        self.omega_mask = None

        self.logs = defaultdict(list)
        
    def create_low_rank_matrix(self):
        """
        M_true (n x m) - random marix
        """
        U = np.random.randn(self.m, self.rank)
        V = np.random.randn(self.n, self.rank)
        self.A = U @ V.T

    def build_omega_mask(self):
        omega_size = self.OS * (self.m + self.n - self.rank) * self.rank
        omega = np.random.choice(m*n, omega_size, replace=False)
        omega_mask = np.zeros((self.m * self.n,), dtype=bool)
        omega_mask[omega] = True
        self.omega_mask = omega_mask.reshape(self.m, self.n)
        self.A_omega = self.A[self.omega_mask]

    def complete_matrix(self):
        """
        Algorithm 1
        """
        self.logs["rank"] = self.rank
        self.logs["m"] = self.m
        self.logs["n"] = self.n

        k = self.rank

        # initialize 0 iteration
        X_L = np.random.randn(self.m, k)
        X_R = np.random.randn(self.n, k)
        X_prev = X_L @ X_R.T
        usv = np.linalg.svd(X_prev)
        U_prev, S_prev, V_prev = usv.U[:, :k], np.diag(usv.S[:k]), usv.Vh.T[:, :k]
        X_omega_prev =  X_prev[self.omega_mask]
        R = X_omega_prev - self.A_omega
        grad_prev = self.riemannian_grad(U_prev, V_prev, R)
        dir_prev = grad_prev

        # Initialize 1st iteration
        X_L = np.random.randn(self.m, k)
        X_R = np.random.randn(self.n, k)
        X = X_L @ X_R.T
        usv = np.linalg.svd(X)
        U, S, V = usv.U[:, :k], np.diag(usv.S[:k]), usv.Vh.T[:, :k]
        X_omega = X[self.omega_mask]

        for i in tqdm(range(self.num_iters), total=self.num_iters):
            R = X_omega - self.A_omega

            grad = self.riemannian_grad(U, V, R)

            if grad.norm <= self.tol:
                break
            
            self.logs["grad_norm"].append(grad.norm)

            dir = self.conjugate_direction(
                U_prev=U_prev, V_prev=V_prev,
                U=U, V=V, grad=grad,
                grad_prev=grad_prev,
                dir_prev=dir_prev,
            )
            self.logs["dir_norm"].append(dir.norm)
            
            X_prev, U_prev, S_prev, V_prev = X, U, S, V
            t_opt = self.initial_guess(U, V, -R, dir)
            X, U, S, V = self.retraction(U, S, V, dir * t_opt*0.5, k)
            X_omega = X[self.omega_mask]

            relative_error = np.linalg.norm(X - self.A) / np.linalg.norm(self.A)
            self.logs["relative_error"].append(relative_error)

            relative_residual = np.linalg.norm(X_omega - self.A_omega) / np.linalg.norm(self.A_omega)
            self.logs["relative_residual"].append(relative_residual)

        return X
    
    def riemannian_grad(self, U, V, R):
        """
        Algorithm 2
        """
        Ru = R.T @ U
        Rv = R @ V

        M = U.T @ Rv

        Up = Rv - U @ M
        Vp = Ru - V @ M.T

        return TangentVector(
            U=U, V=V, M=M,
            Up=Up, Vp=Vp,
        )

    def vector_transport(self, U_plus, V_plus, v):
        """
        Algorithm 3
        """
        Av = v.V.T @ V_plus
        Au = v.U.T @ U_plus

        Bv = v.Vp.T @ V_plus
        Bu = v.Up.T @ U_plus

        M1_plus = Au.T @ v.M @ Av
        U1_plus = v.U @ (v.M @ Av)
        V1_plus = v.V @ (v.M.T @ Au)

        M2_plus = Bu.T @ Av
        U2_plus = v.Up @ Av
        V2_plus = v.V @ Bu

        M3_plus = Au.T @ Bv
        U3_plus = v.U @ Bv
        V3_plus = v.Vp @ Au

        M_plus = M1_plus + M2_plus + M3_plus

        Up_plus = U1_plus + U2_plus + U3_plus
        Up_plus = Up_plus - U_plus @ (U_plus.T @ Up_plus)

        Vp_plus = V1_plus + V2_plus + V3_plus
        Vp_plus = Vp_plus - V_plus @ (V_plus.T @ Vp_plus)

        return TangentVector(
            U=U_plus, V=V_plus, M=M_plus, Up=Up_plus, Vp=Vp_plus
        )

    def conjugate_direction(self, U, V, grad, grad_prev, dir_prev):
        """
        Algorithm 4
        """
        grad_prev_tr = self.vector_transport(U_plus=U, V_plus=V, v=grad_prev)
        dir_prev_tr = self.vector_transport(U_plus=U, V_plus=V, v=dir_prev)

        delta = grad - grad_prev_tr
        beta = max(0, delta.scalar_product(grad) / grad_prev.norm**2)
        dir = dir_prev_tr * beta - grad

        alpha = dir.scalar_product(grad) / (dir.norm * grad.norm)
        if alpha <= 0.1:
            return grad

        return dir

    def initial_guess(self, U, V, R, dir):
        """
        Algorithm 5
        """
        U_ext = np.concatenate([U @ dir.M + dir.Up, U], axis=1)
        V_ext = np.concatenate([V, dir.Vp], axis=1)

        N = self.omega_proj(U_ext @ V_ext.T)
        t_opt = np.trace(N.T @ R) / np.trace(N.T @ N)

        return t_opt

    def retraction(self, U, S, V, scaled_dir, k):
        """
        Algorithm 6
        """
        Qu, Ru = np.linalg.qr(scaled_dir.Up)
        Qv, Rv = np.linalg.qr(scaled_dir.Vp)

        row1 = np.concatenate([S + scaled_dir.M, Rv.T], axis=1)
        row2 = np.concatenate([Ru, np.zeros((Ru.shape[0], row1.shape[1] - Ru.shape[1]))], axis=1)
        s = np.concatenate([row1, row2], axis=0)

        usv = np.linalg.svd(s)
        Us, Ss, Vs = usv.U, usv.S, usv.Vh.T

        eps = np.zeros(k) + 1e-6
        S_plus = np.diag(Ss[:k] + eps)
        U_plus = np.concatenate([U, Qu], axis=1) @ Us[:, :k]
        V_plus = np.concatenate([V, Qv], axis=1) @ Vs[:, :k]

        return U_plus @ S_plus @ V_plus.T, U_plus, S_plus, V_plus

    def run(self):
        """
        Executes pipeline.
        """
        self.build_low_rank_matrix()
        self.build_omega_mask()
        self.complete_matrix()