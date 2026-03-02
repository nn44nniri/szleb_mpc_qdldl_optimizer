# train_szleb_mpcrl_v1.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# gym / gymnasium
try:
    import gymnasium as gym
except ImportError:
    import gym  # type: ignore

from szleb_gym_rl import register_szleb_env, SZLEBEnvConfig
from szleb_mpcrl import MPCRLAgent
from szleb_mpcrl.envs import (
    reset_env,
    step_env,
    get_target_bands_from_env,
    extract_control_state,
)


CSV_PATH = "/media/p1/datasets/weather_Alvand/Alvand_36_186002_50_064982_1640995200_1704067199_67d818b83e2ae2000820e2db.csv"
USE_COLS = ["dt", "dt_iso", "temp", "humidity", "wind_speed", "clouds_all"]

ELAPSED_S_PER_ROW = 3600.0
DT_S = 60.0

DEFAULT_TIN0_C = 20.0
DEFAULT_RHIN0_PCT = 60.0


@dataclass
class TrainCfg:
    episode_len_h: int = 168
    n_train_episodes: int = 300
    n_val_episodes: int = 40
    seed: int = 2026

    lai_start: float = 0.8
    lai_end: float = 2.5

    out_dir: str = "train_logs_weather"
    model_path: str = "szleb_mpc_qdldl_trained.pkl"


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dt_iso_parsed"] = pd.to_datetime(df.get("dt_iso", pd.Series([None] * len(df))), errors="coerce", utc=True)
    if df["dt_iso_parsed"].isna().all() and "dt" in df.columns:
        df["dt_iso_parsed"] = pd.to_datetime(df["dt"], unit="s", utc=True, errors="coerce")
    return df


def _temp_to_celsius(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    med = float(np.nanmedian(s.to_numpy()))
    if med > 60.0:
        return s - 273.15
    return s


def load_weather_csv(path: str) -> pd.DataFrame:
    print(f"[DATA] Loading CSV: {path}")
    df = pd.read_csv(path, usecols=lambda c: c in USE_COLS)
    df = _ensure_datetime(df)
    df = df.dropna(subset=["dt", "temp", "humidity", "wind_speed", "clouds_all"]).reset_index(drop=True)

    df["t_out_c"] = _temp_to_celsius(df["temp"])
    df["rh_out_pct"] = df["humidity"].astype(float).clip(0, 100)

    clouds = df["clouds_all"].astype(float).clip(0, 100)
    df["g_sun_w_m2"] = 400.0 * (1.0 - clouds / 100.0)

    df["wind_speed_m_s"] = df["wind_speed"].astype(float).clip(lower=0)

    if df["dt_iso_parsed"].notna().any():
        df = df.sort_values("dt_iso_parsed").reset_index(drop=True)
        print(f"[DATA] Parsed dt_iso. Range: {df['dt_iso_parsed'].iloc[0]}  ->  {df['dt_iso_parsed'].iloc[-1]}")
    else:
        df = df.sort_values("dt").reset_index(drop=True)
        print(f"[DATA] Using unix dt. Range: {int(df['dt'].iloc[0])} -> {int(df['dt'].iloc[-1])}")

    print(f"[DATA] Rows after cleaning: {len(df)}")
    return df


def split_train_val(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df["dt_iso_parsed"].notna().any():
        year = df["dt_iso_parsed"].dt.year
        month = df["dt_iso_parsed"].dt.month
        df_2022 = df.loc[year == 2022].copy()
        if len(df_2022) > 1000:
            train = df_2022.loc[month <= 10].reset_index(drop=True)
            val = df_2022.loc[month >= 11].reset_index(drop=True)
            if len(train) > 1000 and len(val) > 100:
                print(f"[DATA] Split by month: train={len(train)} (Jan-Oct 2022), val={len(val)} (Nov-Dec 2022)")
                return train, val

    n = len(df)
    cut = int(0.8 * n)
    train = df.iloc[:cut].reset_index(drop=True)
    val = df.iloc[cut:].reset_index(drop=True)
    print(f"[DATA] Split by 80/20: train={len(train)}, val={len(val)}")
    return train, val


def sample_random_window(df: pd.DataFrame, length: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) < length + 1:
        raise ValueError(f"Dataset too short for window length={length}. len(df)={len(df)}")
    start = int(rng.integers(0, len(df) - length))
    return df.iloc[start:start + length].reset_index(drop=True)


def build_season_table(df_window: pd.DataFrame, tin0: float, rh0: float, lai_start: float, lai_end: float) -> pd.DataFrame:
    n = len(df_window)
    lai = np.linspace(lai_start, lai_end, n, dtype=float)
    return pd.DataFrame({
        "day": np.arange(n, dtype=int),
        "LAI": lai,
        "t_out_c": df_window["t_out_c"].to_numpy(dtype=float),
        "rh_out_pct": df_window["rh_out_pct"].to_numpy(dtype=float),
        "g_sun_w_m2": df_window["g_sun_w_m2"].to_numpy(dtype=float),
        "t_in_c": float(tin0),
        "rh_in_pct": float(rh0),
        "wind_speed_m_s": df_window["wind_speed_m_s"].to_numpy(dtype=float),
    })


def make_szleb_env_hourly(season_table: pd.DataFrame):
    register_szleb_env()
    cfg = SZLEBEnvConfig(
        dt_s=DT_S,
        elapsed_s_per_row=ELAPSED_S_PER_ROW,
        use_action_override=True,
    )
    return gym.make("SZLEB-v0", season_table=season_table, config=cfg)


def z_from_row(row: pd.Series) -> np.ndarray:
    return np.array([float(row["t_out_c"]), float(row["rh_out_pct"]), float(row["g_sun_w_m2"]), float(row["LAI"])], dtype=float)


from tqdm import tqdm
import numpy as np

# --- train_szleb_mpcrl_v2.py (replace run_episode) ---
from tqdm import tqdm
import numpy as np

def run_episode(env, agent, season, train: bool, *, episode_idx: int = 0, total_episodes: int = 0):
    obs, info = reset_env(env)

    H = agent.cfg.mpc.horizon
    done = False
    t = 0

    total_reward = 0.0
    total_cost = 0.0

    x = extract_control_state(info, fallback=np.asarray(obs))

    pbar = None
    if train:
        total_steps = len(season) if hasattr(season, "__len__") else None
        desc = f"EP {episode_idx+1}/{total_episodes}" if total_episodes else f"EP {episode_idx+1}"
        pbar = tqdm(total=total_steps, desc=desc, leave=False, dynamic_ncols=True)

    try:
        while not done:
            # horizon disturbances
            z_seq = np.zeros((agent.nz, H), dtype=float)
            for k in range(H):
                idx = min(t + k, len(season) - 1)
                row = season.iloc[idx].to_dict()
                z_seq[:, k] = np.array(
                    [
                        float(row.get("t_out_c", 0.0)),
                        float(row.get("rh_out_pct", 0.0)),
                        float(row.get("g_sun_w_m2", 0.0)),
                        float(row.get("LAI", 0.0)),
                    ],
                    dtype=float,
                )

            u = agent.act(x, z_seq, training=train)

            obs2, reward, done, info2 = step_env(env, u)
            x_next = extract_control_state(info2, fallback=np.asarray(obs2))

            row_now = season.iloc[min(t, len(season) - 1)].to_dict()
            z_now = np.array(
                [
                    float(row_now.get("t_out_c", 0.0)),
                    float(row_now.get("rh_out_pct", 0.0)),
                    float(row_now.get("g_sun_w_m2", 0.0)),
                    float(row_now.get("LAI", 0.0)),
                ],
                dtype=float,
            )

            if train:
                out = agent.learn_from_transition(
                    x=x, u=u, z=z_now, x_next=x_next,
                    info=info2, info_next=None
                )
                total_cost += float(out.get("cost", 0.0))

            total_reward += float(reward)

            if pbar is not None:
                pbar.update(1)
                if (t % 10) == 0:
                    tin = float(x_next[0]) if len(x_next) >= 1 else float("nan")
                    rh = float(x_next[1]) if len(x_next) >= 2 else float("nan")
                    pbar.set_postfix(
                        step=t,
                        Tin=f"{tin:.2f}",
                        RH=f"{rh:.1f}",
                        rew=f"{total_reward:.2f}",
                        cost=f"{total_cost:.2f}",
                        u0=f"{float(u[0]):.2f}" if len(u) else "na",
                    )

            obs, info = obs2, info2
            x = x_next
            t += 1

    finally:
        if pbar is not None:
            pbar.close()

    return {
        "steps": t,
        "reward_total": float(total_reward),
        "cost_ours_total": float(total_cost),
        "w_energy": float(agent.w.w_energy),
        "w_slack_temp": float(agent.w.w_slack_temp),
        "w_slack_rh": float(agent.w.w_slack_rh),
    }

def main():
    cfg = TrainCfg()
    os.makedirs(cfg.out_dir, exist_ok=True)

    df = load_weather_csv(CSV_PATH)
    df_train, df_val = split_train_val(df)

    rng = np.random.default_rng(cfg.seed)

    # Warm env to get action space
    warm_w = sample_random_window(df_train, cfg.episode_len_h, rng)
    warm_season = build_season_table(warm_w, DEFAULT_TIN0_C, DEFAULT_RHIN0_PCT, cfg.lai_start, cfg.lai_end)
    warm_env = make_szleb_env_hourly(warm_season)

    agent = MPCRLAgent(action_space=warm_env.action_space)
    agent.set_targets(get_target_bands_from_env(warm_env))

    train_logs = []
    print("\n[TRAIN] Starting training...")
    for ep in tqdm(range(cfg.n_train_episodes), desc="TRAIN episodes"):
        w = sample_random_window(df_train, cfg.episode_len_h, rng)
        season = build_season_table(w, DEFAULT_TIN0_C, DEFAULT_RHIN0_PCT, cfg.lai_start, cfg.lai_end)
        env = make_szleb_env_hourly(season)

        agent.set_targets(get_target_bands_from_env(env))
        log = run_episode(env, agent, season, train=True)
        log.update({"episode": ep, "split": "train"})
        train_logs.append(log)

    agent.save(cfg.model_path)
    print(f"\n[TRAIN] Saved model: {cfg.model_path}")

    val_logs = []
    print("\n[VAL] Starting validation...")
    for ep in tqdm(range(cfg.n_val_episodes), desc="VAL episodes"):
        w = sample_random_window(df_val, cfg.episode_len_h, rng)
        season = build_season_table(w, DEFAULT_TIN0_C, DEFAULT_RHIN0_PCT, cfg.lai_start, cfg.lai_end)
        env = make_szleb_env_hourly(season)

        eval_agent = MPCRLAgent.load(cfg.model_path, action_space=env.action_space)
        eval_agent.set_targets(get_target_bands_from_env(env))
        eval_agent.freeze_learning()

        log = run_episode(env, eval_agent, season, train=False)
        log.update({"episode": ep, "split": "val"})
        val_logs.append(log)

    pd.DataFrame(train_logs).to_csv(os.path.join(cfg.out_dir, "train_episode_log.csv"), index=False)
    pd.DataFrame(val_logs).to_csv(os.path.join(cfg.out_dir, "val_episode_log.csv"), index=False)

    print(f"\nSaved logs in: {cfg.out_dir}")
    print(" - train_episode_log.csv")
    print(" - val_episode_log.csv")


if __name__ == "__main__":
    main()

