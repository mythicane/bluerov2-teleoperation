"""Load and pool BlueROV2 creo_pool episodes for the perturbation-rejection study.

Pools all 4 rod colors into a single "calm" set and a single "turbulent" ("under
currents") set, per the study design in method.md. Builds a synthetic continuous
clock per condition since the raw data has no absolute timestamp across episodes
(each episode's `timestamp` column resets to 0) -- see report Limitations section.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CREO_POOL_ROOT = Path(__file__).resolve().parents[2]
COLORS = ["green", "red", "white", "yellow"]
FPS = 50.0
DT = 1.0 / FPS
PWM_NEUTRAL = 1500.0

STATE_NAMES = [
    "roll", "pitch", "yaw", "depth", "depth_setpoint", "pid_z",
    "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
    "baro_press_hpa", "enclosure_temp_c",
    "thr1_pwm", "thr2_pwm", "thr3_pwm", "thr4_pwm", "thr5_pwm", "thr6_pwm",
    "rollspeed", "pitchspeed", "yawspeed", "climb",
    "vibration_x", "vibration_y", "vibration_z",
    "ekf_compass_variance", "ekf_pos_horiz_variance", "ekf_pos_vert_variance",
]
STATE_IDX = {name: i for i, name in enumerate(STATE_NAMES)}
ACC_IDX = [STATE_IDX["acc_x"], STATE_IDX["acc_y"], STATE_IDX["acc_z"]]
PWM_IDX = [STATE_IDX[f"thr{i}_pwm"] for i in range(1, 7)]
ROLL_IDX = STATE_IDX["roll"]
PITCH_IDX = STATE_IDX["pitch"]


@dataclass
class PooledDataset:
    condition: str  # "calm" or "turbulent"
    t: np.ndarray  # (N,) synthetic global time [s], per-condition clock
    acc: np.ndarray  # (N, 3)
    roll: np.ndarray  # (N,)
    pitch: np.ndarray  # (N,)
    pwm: np.ndarray  # (N, 6) raw PWM
    episode_uid: np.ndarray  # (N,) int, unique across colors within this condition
    color: np.ndarray  # (N,) object array of color strings
    episode_boundaries: list  # list of (start_idx, end_idx, color, local_episode_idx)
    dt: float = DT
    fps: float = FPS

    @property
    def u(self) -> np.ndarray:
        """Centered thruster command, u(t) = pwm - neutral."""
        return self.pwm - PWM_NEUTRAL

    @property
    def n(self) -> int:
        return self.t.shape[0]


def _dataset_dir(color: str, turbulent: bool) -> Path:
    name = f"grab {color} rod" + (" under currents" if turbulent else "")
    return CREO_POOL_ROOT / name


def _episode_files(color: str, turbulent: bool) -> list[Path]:
    d = _dataset_dir(color, turbulent) / "data" / "chunk-000"
    return sorted(d.glob("episode_*.parquet"))


def load_condition(condition: str) -> PooledDataset:
    """Pool all 4 rod colors for a given condition ('calm' or 'turbulent')."""
    assert condition in ("calm", "turbulent")
    turbulent = condition == "turbulent"

    t_chunks, acc_chunks, roll_chunks, pitch_chunks, pwm_chunks = [], [], [], [], []
    episode_uid_chunks, color_chunks = [], []
    boundaries = []

    cursor = 0  # global frame counter -> synthetic clock
    episode_uid = 0
    for color in COLORS:
        files = _episode_files(color, turbulent)
        for local_idx, f in enumerate(files):
            df = pd.read_parquet(f, columns=["observation.state"])
            state = np.stack(df["observation.state"].values).astype(np.float64)
            n = state.shape[0]

            frame_t = (cursor + np.arange(n)) * DT
            t_chunks.append(frame_t)
            acc_chunks.append(state[:, ACC_IDX])
            roll_chunks.append(state[:, ROLL_IDX])
            pitch_chunks.append(state[:, PITCH_IDX])
            pwm_chunks.append(state[:, PWM_IDX])
            episode_uid_chunks.append(np.full(n, episode_uid, dtype=np.int64))
            color_chunks.append(np.full(n, color, dtype=object))

            boundaries.append((cursor, cursor + n, color, local_idx))
            cursor += n
            episode_uid += 1

    return PooledDataset(
        condition=condition,
        t=np.concatenate(t_chunks),
        acc=np.concatenate(acc_chunks, axis=0),
        roll=np.concatenate(roll_chunks),
        pitch=np.concatenate(pitch_chunks),
        pwm=np.concatenate(pwm_chunks, axis=0),
        episode_uid=np.concatenate(episode_uid_chunks),
        color=np.concatenate(color_chunks),
        episode_boundaries=boundaries,
    )


if __name__ == "__main__":
    for cond in ("calm", "turbulent"):
        ds = load_condition(cond)
        print(cond, "n_frames", ds.n, "n_episodes", len(ds.episode_boundaries),
              "duration_s", ds.n * ds.dt)
