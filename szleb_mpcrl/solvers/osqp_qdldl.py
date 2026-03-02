# szleb_mpcrl/solvers/osqp_qdldl.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp

from .csc_utils import to_upper_csc, clip_box
from .qdldl_ctypes import (
    factor_quasidefinite_upper_csc,
    solve_inplace,
    QDLDLFactor,
    _INT_DTYPE,
    _FLOAT_DTYPE,
)


@dataclass
class OSQPSolverSettings:
    rho: float = 0.1
    sigma: float = 1e-6
    alpha: float = 1.6
    max_iter: int = 400
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4


@dataclass
class OSQPSolution:
    x: np.ndarray
    y: np.ndarray
    status: str
    iter: int
    prim_res: float
    dual_res: float


class OSQP_QDLDL:
    """
    Solve QP:
        minimize 0.5 x^T P x + q^T x
        subject to l <= A x <= u

    Using ADMM with KKT solve via QDLDL (C backend).
    """

    def __init__(self, settings: OSQPSolverSettings = OSQPSolverSettings()):
        self.st = settings
        self._fact: Optional[QDLDLFactor] = None

        self.P: Optional[sp.csc_matrix] = None
        self.A: Optional[sp.csc_matrix] = None
        self.q: Optional[np.ndarray] = None
        self.l: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self._AT: Optional[sp.csc_matrix] = None

    def setup(self, P: sp.csc_matrix, q: np.ndarray, A: sp.csc_matrix, l: np.ndarray, u: np.ndarray) -> None:
        P = P.tocsc()
        A = A.tocsc()

        q = np.asarray(q, dtype=_FLOAT_DTYPE).reshape(-1)
        l = np.asarray(l, dtype=_FLOAT_DTYPE).reshape(-1)
        u = np.asarray(u, dtype=_FLOAT_DTYPE).reshape(-1)

        n = P.shape[0]
        m = A.shape[0]
        if P.shape != (n, n):
            raise ValueError("P must be (n,n).")
        if A.shape[1] != n:
            raise ValueError("A must be (m,n).")
        if q.size != n or l.size != m or u.size != m:
            raise ValueError("q/l/u size mismatch.")

        self.P, self.A, self.q, self.l, self.u = P, A, q, l, u
        self._AT = A.T.tocsc()

        rho = float(self.st.rho)
        sigma = float(self.st.sigma)

        # KKT matrix (quasi-definite):
        # [P + sigma I     A^T]
        # [A            -(1/rho) I]
        K11 = P + sigma * sp.eye(n, format="csc")
        K22 = (-1.0 / rho) * sp.eye(m, format="csc")
        K = sp.bmat([[K11, self._AT], [A, K22]], format="csc")

        K_up = to_upper_csc(K)

        # Enforce ABI types for QDLDL: int64 indices, float64 data (from headers)
        Ap = np.ascontiguousarray(K_up.indptr.astype(_INT_DTYPE, copy=False))
        Ai = np.ascontiguousarray(K_up.indices.astype(_INT_DTYPE, copy=False))
        Ax = np.ascontiguousarray(K_up.data.astype(_FLOAT_DTYPE, copy=False))

        self._fact = factor_quasidefinite_upper_csc(K_up.shape[0], Ap, Ai, Ax)

    def solve(self, x0: Optional[np.ndarray] = None, y0: Optional[np.ndarray] = None) -> OSQPSolution:
        if self.P is None or self.A is None or self.q is None or self.l is None or self.u is None:
            raise RuntimeError("Call setup(P,q,A,l,u) first.")
        if self._fact is None:
            raise RuntimeError("Factorization missing. setup() must be called first.")

        P, A, AT = self.P, self.A, self._AT
        q, l, u = self.q, self.l, self.u
        n = P.shape[0]
        m = A.shape[0]

        rho = float(self.st.rho)
        sigma = float(self.st.sigma)
        alpha = float(self.st.alpha)

        x = np.zeros(n, dtype=_FLOAT_DTYPE) if x0 is None else np.asarray(x0, dtype=_FLOAT_DTYPE).reshape(n).copy()
        y = np.zeros(m, dtype=_FLOAT_DTYPE) if y0 is None else np.asarray(y0, dtype=_FLOAT_DTYPE).reshape(m).copy()
        z = clip_box(A.dot(x), l, u).astype(_FLOAT_DTYPE, copy=False)

        status = "max_iter"
        prim_res = float("inf")
        dual_res = float("inf")

        for k in range(1, int(self.st.max_iter) + 1):
            # rhs for KKT solve
            rhs1 = sigma * x - q
            rhs2 = z - (1.0 / rho) * y
            rhs = np.ascontiguousarray(np.concatenate([rhs1, rhs2]).astype(_FLOAT_DTYPE, copy=False))

            # Solve in-place using C QDLDL
            solve_inplace(self._fact, rhs)

            x_tilde = rhs[:n]
            nu = rhs[n:]  # dual for equality in KKT form

            z_tilde = z + (1.0 / rho) * (nu - y)

            x_new = alpha * x_tilde + (1.0 - alpha) * x
            v = alpha * z_tilde + (1.0 - alpha) * z + (1.0 / rho) * y
            z_new = clip_box(v, l, u).astype(_FLOAT_DTYPE, copy=False)
            y_new = y + rho * (alpha * z_tilde + (1.0 - alpha) * z - z_new)

            r_prim = A.dot(x_new) - z_new
            r_dual = (P.dot(x_new) + q) + AT.dot(y_new)

            prim_res = float(np.max(np.abs(r_prim)))
            dual_res = float(np.max(np.abs(r_dual)))

            eps_prim = float(self.st.eps_abs + self.st.eps_rel * max(np.max(np.abs(A.dot(x_new))), np.max(np.abs(z_new))))
            eps_dual = float(self.st.eps_abs + self.st.eps_rel * max(np.max(np.abs(P.dot(x_new))), np.max(np.abs(AT.dot(y_new))), np.max(np.abs(q))))

            x, z, y = x_new, z_new, y_new

            if prim_res <= eps_prim and dual_res <= eps_dual:
                status = "solved"
                break

        return OSQPSolution(x=x, y=y, status=status, iter=k, prim_res=prim_res, dual_res=dual_res)