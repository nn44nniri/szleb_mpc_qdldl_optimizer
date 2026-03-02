# szleb_mpcrl/mpc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp

from .costs import Weights, TargetBands
from .solvers.osqp_qdldl import OSQP_QDLDL, OSQPSolverSettings


@dataclass
class MPCConfig:
    horizon: int = 12

    # tracking weights (NEW)
    w_track_t: float = 30.0
    w_track_rh: float = 10.0
    w_terminal_mult: float = 2.0

    # smoothness + energy
    smooth_u: float = 0.10

    # solver
    solver_max_iter: int = 400
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4
    rho: float = 0.1
    sigma: float = 1e-6
    alpha: float = 1.6


def _theta_to_lin(theta: np.ndarray, nx: int, nu: int, nz: int):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (nx, nx + nu + nz + 1):
        raise ValueError(f"Theta must be {(nx, nx+nu+nz+1)}, got {theta.shape}")
    A = theta[:, 0:nx]
    B = theta[:, nx:nx + nu]
    E = theta[:, nx + nu:nx + nu + nz]
    b = theta[:, -1]
    return A, B, E, b


def mpc_solve(
    *,
    x0: np.ndarray,
    z_seq: np.ndarray,           # (nz, H)
    u_prev: np.ndarray,
    weights: Weights,
    targets: TargetBands,
    theta: np.ndarray,
    u_low: np.ndarray,
    u_high: np.ndarray,
    cfg: MPCConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Convex QP MPC:
      minimize tracking + energy + smoothing + slack^2
      subject to linear dynamics + band constraints with slacks + bounds
    """

    nx = 2
    x0 = np.asarray(x0, dtype=float).reshape(nx)

    u_prev = np.asarray(u_prev, dtype=float).reshape(-1)
    u_low = np.asarray(u_low, dtype=float).reshape(-1)
    u_high = np.asarray(u_high, dtype=float).reshape(-1)
    nu = int(u_low.size)

    H = int(cfg.horizon)
    z_seq = np.asarray(z_seq, dtype=float)
    nz = int(z_seq.shape[0])
    if z_seq.shape[1] != H:
        raise ValueError(f"z_seq must be (nz,H). Got {z_seq.shape}, H={H}")

    Ahat, Bhat, Ehat, bhat = _theta_to_lin(theta, nx=nx, nu=nu, nz=nz)

    tin_lo, tin_hi = targets.tin_opt_c
    rh_lo, rh_hi = targets.rh_opt_pct
    tin_mid = 0.5 * (tin_lo + tin_hi)
    rh_mid = 0.5 * (rh_lo + rh_hi)

    # decision vector w = [x(0..H), u(0..H-1), sT(0..H-1), sRH(0..H-1)]
    n_x = nx * (H + 1)
    n_u = nu * H
    n_s = 2 * H
    n = n_x + n_u + n_s

    def ix_x(k):  # slice for x_k
        return slice(k * nx, (k + 1) * nx)

    def ix_u(k):  # slice for u_k
        base = n_x + k * nu
        return slice(base, base + nu)

    def ix_sT(k):  # scalar index
        return n_x + n_u + k

    def ix_sRH(k):
        return n_x + n_u + H + k

    # ---- Objective: 0.5 w^T P w + q^T w ----
    P = sp.lil_matrix((n, n), dtype=float)
    q = np.zeros(n, dtype=float)

    w_energy = float(weights.w_energy)
    w_smooth = float(cfg.smooth_u)

    # Make slacks expensive enough to prefer real actuation
    w_sT = float(max(weights.w_slack_temp, 200.0))
    w_sRH = float(max(weights.w_slack_rh, 200.0))

    # Tracking weights
    wT = float(cfg.w_track_t)
    wRH = float(cfg.w_track_rh)

    # tracking cost on x(1..H): (Tin - mid)^2 + (RH - mid)^2
    # implement by expanding: (x - c)^2 = x^2 - 2c x + const
    for k in range(1, H + 1):
        # terminal step gets higher weight
        mult = float(cfg.w_terminal_mult) if k == H else 1.0
        # Tin
        idxT = ix_x(k).start + 0
        P[idxT, idxT] += 2.0 * (wT * mult)
        q[idxT] += -2.0 * (wT * mult) * float(tin_mid)

        # RH
        idxRH = ix_x(k).start + 1
        P[idxRH, idxRH] += 2.0 * (wRH * mult)
        q[idxRH] += -2.0 * (wRH * mult) * float(rh_mid)

    # energy cost on u
    for k in range(H):
        us = ix_u(k)
        for i in range(us.start, us.stop):
            P[i, i] += 2.0 * w_energy

    # smoothness cost
    if w_smooth > 0:
        for k in range(H):
            us = ix_u(k)
            if k == 0:
                for j in range(nu):
                    idx = us.start + j
                    P[idx, idx] += 2.0 * w_smooth
                    q[idx] += -2.0 * w_smooth * float(u_prev[j])
            else:
                u_prev_s = ix_u(k - 1)
                for j in range(nu):
                    i1 = us.start + j
                    i0 = u_prev_s.start + j
                    P[i1, i1] += 2.0 * w_smooth
                    P[i0, i0] += 2.0 * w_smooth
                    P[i1, i0] += -2.0 * w_smooth
                    P[i0, i1] += -2.0 * w_smooth

    # slack quadratics
    for k in range(H):
        P[ix_sT(k), ix_sT(k)] += 2.0 * w_sT
        P[ix_sRH(k), ix_sRH(k)] += 2.0 * w_sRH

    P = P.tocsc()

    # ---- Constraints: l <= A w <= u ----
    rows, cols, data = [], [], []
    l_list, u_list = [], []

    def add_row(ent: Dict[int, float], lv: float, uv: float):
        r = len(l_list)
        for c, v in ent.items():
            rows.append(r); cols.append(c); data.append(float(v))
        l_list.append(float(lv)); u_list.append(float(uv))

    # x(0) == x0
    for i in range(nx):
        add_row({ix_x(0).start + i: 1.0}, x0[i], x0[i])

    # dynamics equalities
    for k in range(H):
        zk = z_seq[:, k].reshape(nz)
        rhs = (Ehat @ zk + bhat).reshape(nx)
        for i in range(nx):
            ent = {ix_x(k + 1).start + i: 1.0}
            for j in range(nx):
                ent[ix_x(k).start + j] = ent.get(ix_x(k).start + j, 0.0) - float(Ahat[i, j])
            for j in range(nu):
                ent[ix_u(k).start + j] = ent.get(ix_u(k).start + j, 0.0) - float(Bhat[i, j])
            add_row(ent, rhs[i], rhs[i])

    # band constraints with slacks on x(k+1)
    for k in range(H):
        Tin_idx = ix_x(k + 1).start + 0
        RH_idx = ix_x(k + 1).start + 1
        sT = ix_sT(k)
        sRH = ix_sRH(k)

        # tin_lo <= Tin + sT  and  Tin - sT <= tin_hi
        add_row({Tin_idx: 1.0, sT: 1.0}, tin_lo, np.inf)
        add_row({Tin_idx: 1.0, sT: -1.0}, -np.inf, tin_hi)

        # rh_lo <= RH + sRH  and  RH - sRH <= rh_hi
        add_row({RH_idx: 1.0, sRH: 1.0}, rh_lo, np.inf)
        add_row({RH_idx: 1.0, sRH: -1.0}, -np.inf, rh_hi)

        # slacks >= 0
        add_row({sT: 1.0}, 0.0, np.inf)
        add_row({sRH: 1.0}, 0.0, np.inf)

    # u bounds
    for k in range(H):
        for j in range(nu):
            idx = ix_u(k).start + j
            add_row({idx: 1.0}, u_low[j], u_high[j])

    m = len(l_list)
    A_qp = sp.coo_matrix((data, (rows, cols)), shape=(m, n)).tocsc()
    l = np.asarray(l_list, dtype=float)
    u = np.asarray(u_list, dtype=float)

    # ---- Solve ----
    solver = OSQP_QDLDL(
        settings=OSQPSolverSettings(
            rho=float(cfg.rho),
            sigma=float(cfg.sigma),
            alpha=float(cfg.alpha),
            max_iter=int(cfg.solver_max_iter),
            eps_abs=float(cfg.eps_abs),
            eps_rel=float(cfg.eps_rel),
        )
    )
    solver.setup(P=P, q=q, A=A_qp, l=l, u=u)
    sol = solver.solve()
    w_opt = sol.x

    u0 = np.asarray(w_opt[ix_u(0)], dtype=float).reshape(nu)
    mpc_diag = {"sT0": float(w_opt[ix_sT(0)]) if H else 0.0,
                "sRH0": float(w_opt[ix_sRH(0)]) if H else 0.0,
                "qp_status": sol.status,
                "qp_iter": sol.iter}
    return u0, mpc_diag