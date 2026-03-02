# szleb_mpcrl/costs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
import numpy as np



@dataclass
class TargetBands:
    # Required / used by MPC now
    tin_opt_c: Tuple[float, float] = (24.0, 26.0)
    rh_opt_pct: Tuple[float, float] = (60.0, 80.0)

    # Optional fields for compatibility with evaluate_* scripts and env configs
    vpd_opt_kpa: Optional[Tuple[float, float]] = None
    co2_opt_ppm: Optional[Tuple[float, float]] = None
    par_opt_umol_m2_s: Optional[Tuple[float, float]] = None
    srad_opt_w_m2: Optional[Tuple[float, float]] = None
    dli_opt_mol_m2_d: Optional[Tuple[float, float]] = None

    @classmethod
    def from_env_config(cls, cfg: Any) -> "TargetBands":
        """
        Compatibility helper expected by envs.py:
            TargetBands.from_env_config(cfg)
        Also supports optional bands if present in cfg.
        """
        def pick(container: Any, keys, default=None):
            if container is None:
                return default
            if isinstance(container, dict):
                for k in keys:
                    if k in container:
                        return container[k]
                return default
            for k in keys:
                if hasattr(container, k):
                    return getattr(container, k)
            return default

        nested = pick(cfg, ["targets", "optimal", "bands", "target_bands"], default=None)

        tin = pick(nested, ["tin_opt_c", "Tin_opt_c", "tin_band_c", "Tin_band_c"], default=None)
        rh = pick(nested, ["rh_opt_pct", "RH_opt_pct", "rh_band_pct", "RH_band_pct"], default=None)

        vpd = pick(nested, ["vpd_opt_kpa", "VPD_opt_kpa", "vpd_band_kpa"], default=None)
        co2 = pick(nested, ["co2_opt_ppm", "CO2_opt_ppm", "co2_band_ppm"], default=None)
        par = pick(nested, ["par_opt_umol_m2_s", "PAR_opt_umol_m2_s", "par_band_umol_m2_s"], default=None)
        srad = pick(nested, ["srad_opt_w_m2", "SRAD_opt_w_m2", "srad_band_w_m2"], default=None)
        dli = pick(nested, ["dli_opt_mol_m2_d", "DLI_opt_mol_m2_d", "dli_band_mol_m2_d"], default=None)

        # fall back to flat
        if tin is None:
            tin = pick(cfg, ["tin_opt_c", "Tin_opt_c", "tin_band_c", "Tin_band_c"], default=cls.tin_opt_c)
        if rh is None:
            rh = pick(cfg, ["rh_opt_pct", "RH_opt_pct", "rh_band_pct", "RH_band_pct"], default=cls.rh_opt_pct)

        def norm_pair(x):
            if x is None:
                return None
            lo, hi = map(float, x)
            return (lo, hi) if lo <= hi else (hi, lo)

        tin = norm_pair(tin)
        rh = norm_pair(rh)

        return cls(
            tin_opt_c=tin,
            rh_opt_pct=rh,
            vpd_opt_kpa=norm_pair(vpd),
            co2_opt_ppm=norm_pair(co2),
            par_opt_umol_m2_s=norm_pair(par),
            srad_opt_w_m2=norm_pair(srad),
            dli_opt_mol_m2_d=norm_pair(dli),
        )

    # --- violations used by MPC / cost ---
    def violation_tin(self, tin_c: float) -> float:
        lo, hi = self.tin_opt_c
        if tin_c < lo:
            return float(lo - tin_c)
        if tin_c > hi:
            return float(tin_c - hi)
        return 0.0

    def violation_rh(self, rh_pct: float) -> float:
        lo, hi = self.rh_opt_pct
        if rh_pct < lo:
            return float(lo - rh_pct)
        if rh_pct > hi:
            return float(rh_pct - hi)
        return 0.0


@dataclass
class Weights:
    """
    Band-first philosophy:
      - slack weights dominate (constraint satisfaction)
      - energy is secondary (least energy while satisfying bands)
    """
    # learning params
    gamma: float = 0.98
    td_lr: float = 1e-4

    # stage weights used by TD/diagnostics
    w_temp: float = 1.0
    w_rh: float = 1.0
    w_energy: float = 0.02

    # IMPORTANT: large -> MPC will act to return inside bands
    w_slack_temp: float = 5_000.0
    w_slack_rh: float = 2_000.0


def _get(info: Dict[str, Any], keys, default=0.0) -> float:
    for k in keys:
        if k in info:
            try:
                return float(info[k])
            except Exception:
                pass
    return float(default)


def compute_stage_cost_from_info(info: Dict[str, Any], targets: TargetBands):
    """
    Used by learning/reporting. Keep key flexibility so you don't break logging.
    """
    tin = _get(info, ["Tin", "Tin_C", "T_in", "T_in_c"], default=np.nan)
    rh = _get(info, ["RHin", "RH_in", "RH_in_pct"], default=np.nan)

    e_kwh_eq = _get(info, ["energy_total_kwh_eq", "energy_kwh_eq", "E_kWh_eq"], default=0.0)

    tv = targets.violation_tin(tin) if np.isfinite(tin) else 0.0
    rv = targets.violation_rh(rh) if np.isfinite(rh) else 0.0

    # band-first cost
    cost = (5_000.0 * (tv ** 2)) + (2_000.0 * (rv ** 2)) + (0.02 * e_kwh_eq)

    diag = {
        "temp_violation": float(tv),
        "rh_violation": float(rv),
        "energy_kwh_eq": float(e_kwh_eq),
    }
    return float(cost), diag