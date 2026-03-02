# szleb_mpcrl/agent.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional
import pickle

import numpy as np

from .costs import Weights, TargetBands, compute_stage_cost_from_info
from .mpc import MPCConfig, mpc_solve
from .rls import RLS, RLSConfig


@dataclass
class MPCRLConfig:
    mpc: MPCConfig = field(default_factory=MPCConfig)
    rls: RLSConfig = field(default_factory=RLSConfig)
    nz: int = 4

    # exploration during training
    explore_std: float = 0.10
    explore_prob: float = 0.30

    # optional bounds for future weight adaptation (kept for compatibility)
    w_min: float = 1e-6
    w_max: float = 1e6


def _bounds_from_action_space(action_space):
    low = np.asarray(action_space.low, dtype=float).reshape(-1)
    high = np.asarray(action_space.high, dtype=float).reshape(-1)
    return low, high


class MPCRLAgent:
    def __init__(self, action_space, obs_dim: Optional[int] = None, cfg: MPCRLConfig = MPCRLConfig()):
        self.cfg = cfg
        self.obs_dim = obs_dim

        self.u_low, self.u_high = _bounds_from_action_space(action_space)
        self.nu = int(self.u_low.size)

        self.nx = 2
        self.nz = int(cfg.nz)

        self.rls = RLS(nx=self.nx, nu=self.nu, nz=self.nz, cfg=cfg.rls)

        self.w = Weights()
        # keep energy weight small enough that control is used
        self.w.w_energy = float(max(self.w.w_energy, 0.05))
        # slacks should not be cheap
        self.w.w_slack_temp = float(max(self.w.w_slack_temp, 200.0))
        self.w.w_slack_rh = float(max(self.w.w_slack_rh, 200.0))

        self.targets: Optional[TargetBands] = None
        self.u_prev = np.zeros(self.nu, dtype=float)

        self._last_mpc_diag: Dict[str, float] = {"sT0": 0.0, "sRH0": 0.0}

        self._rng = np.random.default_rng(2026)

        # NEW: support freezing learning (needed by train_szleb_mpcrl_v2.py)
        self._frozen = False

    # ---------------- targets / mode ----------------
    def set_targets(self, targets: TargetBands) -> None:
        self.targets = targets

    def freeze_learning(self) -> None:
        """Disable online learning updates (RLS/weights). Useful for evaluation."""
        self._frozen = True

    def unfreeze_learning(self) -> None:
        """Enable online learning updates."""
        self._frozen = False

    def is_frozen(self) -> bool:
        return bool(self._frozen)

    # ---------------- policy (MPC) ----------------
    def act(self, x: np.ndarray, z_seq: np.ndarray, *, training: bool = False) -> np.ndarray:
        if self.targets is None:
            raise RuntimeError("Targets not set. Call agent.set_targets(...).")

        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size < 2:
            raise ValueError("x must contain [Tin, RHin] in first 2 values.")
        x2 = np.array([float(x[0]), float(x[1])], dtype=float)

        u, diag = mpc_solve(
            x0=x2,
            z_seq=np.asarray(z_seq, dtype=float),
            u_prev=self.u_prev,
            weights=self.w,
            targets=self.targets,
            theta=self.rls.Theta,
            u_low=self.u_low,
            u_high=self.u_high,
            cfg=self.cfg.mpc,
        )

        # exploration ONLY during training and ONLY if learning is not frozen
        if (not self._frozen) and training and (self.cfg.explore_std > 0) and (self._rng.random() < self.cfg.explore_prob):
            noise = self._rng.normal(0.0, self.cfg.explore_std, size=u.shape)
            u = np.clip(u + noise, self.u_low, self.u_high)

        self.u_prev = u.copy()
        self._last_mpc_diag = dict(diag)
        return u

    # ---------------- training API expected by trainer ----------------
    def learn_from_transition(
        self,
        *,
        x: np.ndarray,
        u: np.ndarray,
        z: np.ndarray,
        x_next: np.ndarray,
        info: Dict[str, Any],
        info_next: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.update(x=x, u=u, z=z, x_next=x_next, info=info, info_next=info_next)

    def update(
        self,
        *,
        x: np.ndarray,
        u: np.ndarray,
        z: np.ndarray,
        x_next: np.ndarray,
        info: Dict[str, Any],
        info_next: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.targets is None:
            raise RuntimeError("Targets not set.")

        x = np.asarray(x, dtype=float).reshape(-1)
        x_next = np.asarray(x_next, dtype=float).reshape(-1)
        if x.size < 2 or x_next.size < 2:
            raise ValueError("x and x_next must contain Tin,RHin as first 2 entries.")

        x2 = np.array([float(x[0]), float(x[1])], dtype=float)
        x2_next = np.array([float(x_next[0]), float(x_next[1])], dtype=float)

        u = np.asarray(u, dtype=float).reshape(self.nu)
        z = np.asarray(z, dtype=float).reshape(self.nz)

        # Only update RLS if not frozen
        if not self._frozen:
            self.rls.update(x=x2, u=u, z=z, x_next=x2_next)

        # compute cost for logging (works in train + eval)
        cost_now, diag_now = compute_stage_cost_from_info(info, self.targets)
        diag_now = dict(diag_now)
        diag_now.update(self._last_mpc_diag)

        return {
            "cost": float(cost_now),
            "diag": diag_now,
            "weights": self.w,
            "mpc_diag": dict(self._last_mpc_diag),
            "frozen": bool(self._frozen),
        }

    # ---------------- persistence ----------------
    def save(self, path: str) -> None:
        payload = {
            "cfg": asdict(self.cfg),
            "obs_dim": self.obs_dim,
            "weights": asdict(self.w),
            "u_prev": self.u_prev,
            "frozen": self._frozen,
            "rls": {"Theta": self.rls.Theta, "P": self.rls.P, "cfg": asdict(self.rls.cfg)},
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(path: str, action_space, obs_dim: Optional[int] = None) -> "MPCRLAgent":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        cfg_raw = payload["cfg"]
        cfg = MPCRLConfig(
            mpc=MPCConfig(**cfg_raw["mpc"]),
            rls=RLSConfig(**cfg_raw["rls"]),
            nz=int(cfg_raw.get("nz", 4)),
            explore_std=float(cfg_raw.get("explore_std", 0.10)),
            explore_prob=float(cfg_raw.get("explore_prob", 0.30)),
            w_min=float(cfg_raw.get("w_min", 1e-6)),
            w_max=float(cfg_raw.get("w_max", 1e6)),
        )

        agent = MPCRLAgent(action_space=action_space, obs_dim=obs_dim or payload.get("obs_dim"), cfg=cfg)
        agent.w = Weights(**payload["weights"])
        agent.u_prev = np.asarray(payload["u_prev"], dtype=float)

        agent.rls.Theta = np.asarray(payload["rls"]["Theta"], dtype=float)
        agent.rls.P = np.asarray(payload["rls"]["P"], dtype=float)

        agent._frozen = bool(payload.get("frozen", False))
        return agent