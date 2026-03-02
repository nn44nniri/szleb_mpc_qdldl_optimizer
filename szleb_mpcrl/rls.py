# szleb_mpcrl/rls.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RLSConfig:
    lam: float = 0.995
    delta: float = 50.0
    prior_cov_scale: float = 1.0


class RLS:
    """
    RLS for:
        x_{k+1} = A x_k + B u_k + E z_k + b
    phi = [x, u, z, 1]
    """

    def __init__(self, nx: int, nu: int, nz: int, cfg: RLSConfig = RLSConfig()):
        self.nx, self.nu, self.nz = nx, nu, nz
        self.cfg = cfg

        self.nphi = nx + nu + nz + 1
        self.Theta = np.zeros((nx, self.nphi), dtype=float)

        # ---- Simple sign-correct prior so MPC has control authority immediately ----
        A = np.eye(nx)

        B = np.zeros((nx, nu), dtype=float)
        if nu >= 1:
            B[0, 0] = +1.5   # heater warms
            B[1, 0] = -2.0   # heater lowers RH% slightly
        if nu >= 2:
            B[0, 1] = -1.0   # fan cools
            B[1, 1] = -3.0   # fan dehumidifies
        if nu >= 3:
            B[0, 2] = -1.2   # vents cool
            B[1, 2] = -4.0   # vents exchange humidity

        E = np.zeros((nx, nz), dtype=float)
        if nz >= 1:
            E[0, 0] = +0.15  # Tout influences Tin
        if nz >= 2:
            E[1, 1] = +0.10  # RHout influences RHin
        if nz >= 3:
            E[0, 2] = +0.002 # solar heats (proxy)
        if nz >= 4:
            E[1, 3] = +0.8   # LAI → humidity (transpiration proxy)

        b = np.zeros(nx, dtype=float)

        self.Theta[:, 0:nx] = A
        self.Theta[:, nx:nx + nu] = B
        self.Theta[:, nx + nu:nx + nu + nz] = E
        self.Theta[:, -1] = b

        # Covariance
        self.P = np.eye(self.nphi, dtype=float) * float(cfg.delta)

        # Prior confidence (smaller diagonal => stronger prior)
        base = float(cfg.prior_cov_scale)
        for i in range(nx + nu + nz + 1):
            self.P[i, i] = max(1e-6, base)

    def predict(self, *, x: np.ndarray, u: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Predict x_{k+1} using the current Theta estimate.
        This is required by evaluate_szleb_mpcrl_report_v4.py.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        z = np.asarray(z, dtype=float).reshape(-1)

        if x.size < self.nx:
            raise ValueError(f"x must have at least {self.nx} elements.")
        if u.size != self.nu:
            raise ValueError(f"u must have {self.nu} elements.")
        if z.size != self.nz:
            raise ValueError(f"z must have {self.nz} elements.")

        x2 = x[: self.nx]
        phi = np.concatenate([x2, u, z, np.ones(1, dtype=float)]).reshape(self.nphi, 1)
        return (self.Theta @ phi).reshape(self.nx)

    def update(self, x: np.ndarray, u: np.ndarray, z: np.ndarray, x_next: np.ndarray) -> None:
        lam = float(self.cfg.lam)

        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        z = np.asarray(z, dtype=float).reshape(-1)
        x_next = np.asarray(x_next, dtype=float).reshape(-1)

        if x.size < self.nx or x_next.size < self.nx:
            raise ValueError(f"x and x_next must have at least {self.nx} elements.")
        if u.size != self.nu:
            raise ValueError(f"u must have {self.nu} elements.")
        if z.size != self.nz:
            raise ValueError(f"z must have {self.nz} elements.")

        x2 = x[: self.nx]
        x2_next = x_next[: self.nx]

        phi = np.concatenate([x2, u, z, np.ones(1, dtype=float)]).reshape(-1, 1)

        Pphi = self.P @ phi
        denom = lam + float((phi.T @ Pphi).item())
        K = Pphi / denom

        yhat = (self.Theta @ phi).reshape(self.nx)
        err = (x2_next - yhat).reshape(self.nx, 1)

        self.Theta = self.Theta + err @ K.T
        self.P = (self.P - K @ (phi.T @ self.P)) / lam