"""
BlueROV2 ACT policy deployment with DAgger-style data collection.

Modes
-----
  IDLE      (default)  Xbox controller drives the robot; no recording.
  POLICY    (press A)  Episode started; ACT policy drives the robot.
                       Sticks > INTERVENTION_THRESH -> HUMAN for that frame.
  HUMAN                Episode ongoing; human override active.
                       Release sticks -> POLICY resumes.
  X (success) or B (failure) -> save episode, return to IDLE.

Button layout
-------------
  Left  stick Y     forward / back
  Left  stick X     strafe left / right
  Right stick X     yaw
  RT    (axis 5)    gripper close (hold) / open (release)  [always active]
  LB    (button 4)  depth setpoint deeper                  [always active]
  RB    (button 5)  depth setpoint shallower               [always active]
  D-pad up/down     camera tilt (hold)                     [always active]
  D-pad left/right  lights dim / brighten (hold)           [always active]
  Start (button 7)  arm
  Back  (button 6)  disarm
  A     (button 0)  start DAgger episode (IDLE only)
  X     (button 2)  save episode as SUCCESS (recording only)
  B     (button 1)  save episode as FAILURE (recording only)

50 Hz main loop; inference in background thread (~1 Hz) feeding a temporal
ensembler so the robot gets fresh commands every tick.

Usage
-----
  conda activate bluerov
  python deployment_scripts/deploy_bluerov.py \\
      --checkpoint models/bluerov-greenrod-020000.zip \\
      --task-name grab-green-rod \\
      --n-action-steps 25          # optional: execute only first N steps of each chunk
      --ensemble-coeff 0.01        # optional: temporal ensembling decay (default: from policy config)
"""

import argparse
import json
import queue
import sys
import threading
import time
import zipfile
from collections import deque
from pathlib import Path

import subprocess
import av
import numpy as np
import pygame
import requests
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BLUEROV_URL, DATA_ROOT

# ── constants ──────────────────────────────────────────────────────────────────

SEND_HZ             = 50
GAIN                = 0.4
ROT_GAIN            = 1.0
POLICY_GAIN         = 1.9
DEADBAND            = 0.005
TRIGGER_THRESHOLD   = 0.5
CAM_TILT_STEP       = 15
LIGHT_STEP          = 15
LIGHT_MIN           = 1100
LIGHT_MAX           = 1900
INTERVENTION_THRESH = 0.3   # stick deflection that counts as human intervention

# depth PID (matches controller_teleop.py training setup)
DEPTH_RATE = 0.5    # m/s setpoint adjustment rate via bumpers
KP_DEPTH   = 200.0
KI_DEPTH   = 0.0
KD_DEPTH   = 0.0
DEPTH_GAIN = 0.5
DEPTH_MIN  = -0.5
DEPTH_MAX  = 50.0
CAM_W, CAM_H = 854, 480
HUD_H = 200
WIN_W, WIN_H = CAM_W, CAM_H + HUD_H

AX_LEFT_X  = 0; AX_LEFT_Y  = 1; AX_RIGHT_X = 2; AX_RIGHT_Y = 3
AX_LT = 4; AX_RT = 5
BTN_A = 0; BTN_B = 1; BTN_X = 2; BTN_Y = 3
BTN_LB = 4; BTN_RB = 5; BTN_BACK = 6; BTN_START = 7

REPO_ROOT = Path(__file__).parent.parent
SDP       = str(REPO_ROOT / "stream.sdp")
MAV       = f"{BLUEROV_URL}/mavlink2rest"
HEADER    = {"system_id": 255, "component_id": 0, "sequence": 0}

RAW_IMU_MSG_ID = 27       # MAVLink common message ID for RAW_IMU
RAW_IMU_HZ     = 20       # requested stream rate — default is ~2Hz, too sparse for SEND_HZ=50 recording
MG_TO_MS2      = 9.80665 / 1000.0   # RAW_IMU accel is milli-g
MRAD_TO_RADS   = 1.0 / 1000.0       # RAW_IMU gyro is milli-rad/s

# ── health / power alert thresholds ──────────────────────────────────────────────
# BATT_WARN_V / BATT_CRIT_V assume a stock 4S Li-ion pack (~16.8V full) —
# adjust to your battery's datasheet if different.
BATT_WARN_V           = 14.0
BATT_CRIT_V           = 13.2
VCC_WARN_MV           = 4700   # autopilot 5V rail brownout warning
POWER_OVERCURRENT_BITS = 0b11000  # MAV_POWER_STATUS: PERIPH_OVERCURRENT(8) | PERIPH_HIPOWER_OVERCURRENT(16)
_SENSOR_BIT_NAMES = {1: "gyro", 2: "accel", 4: "mag", 8: "baro"}  # MAV_SYS_STATUS_SENSOR (low bits)
_SENSOR_DECODED_MASK = 1 | 2 | 4 | 8


def _sensor_alert_text(telem):
    """Flag sensors that are present+enabled but reporting unhealthy (SYS_STATUS bitmasks)."""
    unhealthy = (telem["sensors_enabled"] & telem["sensors_present"]
                 & (~telem["sensors_health"] & 0xFFFFFFFF))
    if not unhealthy:
        return ""
    names = [name for bit, name in _SENSOR_BIT_NAMES.items() if unhealthy & bit]
    extra = unhealthy & ~_SENSOR_DECODED_MASK & 0xFFFFFFFF
    label = "/".join(names)
    if extra:
        label = f"{label}+0x{extra:x}" if label else f"0x{extra:x}"
    return f"SENSOR UNHEALTHY: {label}"


def compute_alerts(telem):
    """Battery / power-rail / sensor-health warnings, checked every telemetry tick."""
    alerts = []
    if telem["batt_v"] > 0 and telem["batt_v"] <= BATT_CRIT_V:
        alerts.append(f"BATTERY CRITICAL {telem['batt_v']:.1f}V")
    elif telem["batt_v"] > 0 and telem["batt_v"] <= BATT_WARN_V:
        alerts.append(f"battery low {telem['batt_v']:.1f}V")
    if telem["vcc_mv"] and telem["vcc_mv"] < VCC_WARN_MV:
        alerts.append(f"5V rail low {telem['vcc_mv'] / 1000:.2f}V")
    sensor_txt = _sensor_alert_text(telem)
    if sensor_txt:
        alerts.append(sensor_txt)
    if telem["power_flags"] & POWER_OVERCURRENT_BITS:
        alerts.append("OVERCURRENT")
    return alerts


# ── MAVLink / robot interface ──────────────────────────────────────────────────

_session      = requests.Session()
_mc_queue:    queue.Queue = queue.Queue(maxsize=1)
_servo_queue: queue.Queue = queue.Queue(maxsize=8)


def _enqueue_mc(msg):
    try:
        _mc_queue.put_nowait(msg)
    except queue.Full:
        try: _mc_queue.get_nowait()
        except queue.Empty: pass
        _mc_queue.put_nowait(msg)


def _enqueue_servo(msg):
    try: _servo_queue.put_nowait(msg)
    except queue.Full: pass


def _sender_loop(stop):
    while not stop.is_set():
        while True:
            try:
                msg = _servo_queue.get_nowait()
                _session.post(f"{MAV}/mavlink",
                              json={"header": HEADER, "message": msg}, timeout=0.1)
            except (queue.Empty, Exception): break
        try:
            msg = _mc_queue.get(timeout=0.02)
            _session.post(f"{MAV}/mavlink",
                          json={"header": HEADER, "message": msg}, timeout=0.1)
        except (queue.Empty, Exception): pass


def _post_sync(msg):
    try:
        return _session.post(
            f"{MAV}/mavlink", json={"header": HEADER, "message": msg}, timeout=0.5
        ).ok
    except Exception: return False


def _param_id_chars(name):
    return list(name.ljust(16, "\x00")[:16])


def _set_servo_function(servo, value):
    _post_sync({"type": "PARAM_SET", "target_system": 1, "target_component": 1,
                "param_id": _param_id_chars(f"SERVO{servo}_FUNCTION"),
                "param_value": float(value),
                "param_type": {"type": "MAV_PARAM_TYPE_INT8"}})


def _set_motor_direction(motor, direction):
    _post_sync({"type": "PARAM_SET", "target_system": 1, "target_component": 1,
                "param_id": _param_id_chars(f"MOT_{motor}_DIRECTION"),
                "param_value": float(direction),
                "param_type": {"type": "MAV_PARAM_TYPE_INT8"}})


def _set_message_interval(msg_id, hz):
    """MAV_CMD_SET_MESSAGE_INTERVAL: request the autopilot stream msg_id at hz."""
    _post_sync({"type": "COMMAND_LONG", "target_system": 1, "target_component": 1,
                "command": {"type": "MAV_CMD_SET_MESSAGE_INTERVAL"}, "confirmation": 0,
                "param1": float(msg_id), "param2": float(1_000_000 / hz),
                "param3": 0.0, "param4": 0.0, "param5": 0.0, "param6": 0.0, "param7": 0.0})


def _do_set_servo(channel, pwm):
    _enqueue_servo({"type": "COMMAND_LONG", "target_system": 1, "target_component": 1,
                    "command": {"type": "MAV_CMD_DO_SET_SERVO"}, "confirmation": 0,
                    "param1": float(channel), "param2": float(pwm),
                    "param3": 0.0, "param4": 0.0, "param5": 0.0,
                    "param6": 0.0, "param7": 0.0})


def send_heartbeat():
    _post_sync({"type": "HEARTBEAT", "custom_mode": 0,
                "mavtype": {"type": "MAV_TYPE_GCS"},
                "autopilot": {"type": "MAV_AUTOPILOT_INVALID"},
                "base_mode": {"bits": 0},
                "system_status": {"type": "MAV_STATE_ACTIVE"},
                "mavlink_version": 3})


def set_arm(armed):
    return _post_sync({"type": "COMMAND_LONG", "target_system": 1, "target_component": 1,
                       "command": {"type": "MAV_CMD_COMPONENT_ARM_DISARM"},
                       "confirmation": 0,
                       "param1": 1.0 if armed else 0.0, "param2": 21196.0,
                       "param3": 0.0, "param4": 0.0,
                       "param5": 0.0, "param6": 0.0, "param7": 0.0})


def send_manual_control(x, y, z, r):
    _enqueue_mc({"type": "MANUAL_CONTROL", "target": 1,
                 "x": x, "y": y, "z": z, "r": r, "buttons": 0})


def set_gripper(open_gripper):
    _do_set_servo(11, 1900.0 if open_gripper else 1100.0)


def set_camera_tilt(pwm):
    _do_set_servo(16, float(pwm))


def set_lights(pwm):
    _do_set_servo(13, float(pwm))
    _do_set_servo(14, float(pwm))


def telemetry_loop(data, stop):
    while not stop.is_set():
        try:
            att = _session.get(
                f"{MAV}/mavlink/vehicles/1/components/1/messages/ATTITUDE",
                timeout=0.5).json()["message"]
            data["roll"]  = att.get("roll", 0)
            data["pitch"] = att.get("pitch", 0)
            data["yaw"]   = att.get("yaw", 0)
            # EKF-fused angular rates (distinct from RAW_IMU's raw body-frame gyro)
            data["rollspeed"]  = att.get("rollspeed", 0)
            data["pitchspeed"] = att.get("pitchspeed", 0)
            data["yawspeed"]   = att.get("yawspeed", 0)
            p2 = _session.get(
                f"{MAV}/mavlink/vehicles/1/components/1/messages/SCALED_PRESSURE2",
                timeout=0.5).json()["message"]
            data["depth"] = (p2.get("press_abs", 1013.25) - 1013.25) / 98.0
            sys = _session.get(
                f"{MAV}/mavlink/vehicles/1/components/1/messages/SYS_STATUS",
                timeout=0.5).json()["message"]
            data["batt_v"] = sys.get("voltage_battery", 0) / 1000.0
            data["sensors_enabled"] = sys.get("onboard_control_sensors_enabled", {}).get("bits", 0)
            data["sensors_health"]  = sys.get("onboard_control_sensors_health", {}).get("bits", 0)
            data["sensors_present"] = sys.get("onboard_control_sensors_present", {}).get("bits", 0)
            batt = _session.get(
                f"{MAV}/mavlink/vehicles/1/components/1/messages/BATTERY_STATUS",
                timeout=0.5).json()["message"]
            data["batt_mah"] = batt.get("current_consumed", 0)
            imu = _session.get(
                f"{MAV}/mavlink/vehicles/1/components/1/messages/RAW_IMU",
                timeout=0.5).json()["message"]
            # onboard IMU: accel in milli-g -> m/s^2, gyro in milli-rad/s -> rad/s (body frame)
            data["acc_x"]  = imu.get("xacc", 0)  * MG_TO_MS2
            data["acc_y"]  = imu.get("yacc", 0)  * MG_TO_MS2
            data["acc_z"]  = imu.get("zacc", 0)  * MG_TO_MS2
            data["gyro_x"] = imu.get("xgyro", 0) * MRAD_TO_RADS
            data["gyro_y"] = imu.get("ygyro", 0) * MRAD_TO_RADS
            data["gyro_z"] = imu.get("zgyro", 0) * MRAD_TO_RADS
            sp = _session.get(
                f"{MAV}/mavlink/vehicles/1/components/1/messages/SCALED_PRESSURE",
                timeout=0.5).json()["message"]
            # internal enclosure barometer (distinct from SCALED_PRESSURE2, the external depth sensor)
            data["baro_press_hpa"]   = sp.get("press_abs", 0.0)
            data["enclosure_temp_c"] = sp.get("temperature", 0) / 100.0
            sor = _session.get(
                f"{MAV}/mavlink/vehicles/1/components/1/messages/SERVO_OUTPUT_RAW",
                timeout=0.5).json()["message"]
            # BlueROV2 (standard configuration): 6 vectored thrusters on servo1..servo6, raw PWM (neutral ~1500)
            data["thr_pwm"] = [sor.get(f"servo{i}_raw", 1500) for i in range(1, 7)]
            ps = _session.get(
                f"{MAV}/mavlink/vehicles/1/components/1/messages/POWER_STATUS",
                timeout=0.5).json()["message"]
            data["vcc_mv"]      = ps.get("Vcc", 0)
            data["power_flags"] = ps.get("flags", {}).get("bits", 0)
            hud = _session.get(
                f"{MAV}/mavlink/vehicles/1/components/1/messages/VFR_HUD",
                timeout=0.5).json()["message"]
            data["climb"] = hud.get("climb", 0.0)
            vib = _session.get(
                f"{MAV}/mavlink/vehicles/1/components/1/messages/VIBRATION",
                timeout=0.5).json()["message"]
            data["vibration_x"] = vib.get("vibration_x", 0.0)
            data["vibration_y"] = vib.get("vibration_y", 0.0)
            data["vibration_z"] = vib.get("vibration_z", 0.0)
            ekf = _session.get(
                f"{MAV}/mavlink/vehicles/1/components/1/messages/EKF_STATUS_REPORT",
                timeout=0.5).json()["message"]
            data["ekf_compass_variance"]    = ekf.get("compass_variance", 0.0)
            data["ekf_pos_horiz_variance"]  = ekf.get("pos_horiz_variance", 0.0)
            data["ekf_pos_vert_variance"]   = ekf.get("pos_vert_variance", 0.0)
            # airspeed_variance/terrain_alt_variance/velocity_variance dropped: confirmed
            # hard-zero on this vehicle (no airspeed sensor -- N/A underwater; no
            # rangefinder/altimeter; no GPS or DVL for aided velocity)
        except Exception: pass
        time.sleep(0.2)


FFMPEG = "ffmpeg"


def camera_loop(shared, stop):
    frame_bytes = CAM_W * CAM_H * 3
    corrupt = threading.Event()
    cmd = [
        FFMPEG,
        "-loglevel", "warning",
        "-protocol_whitelist", "file,udp,rtp,crypto",
        "-buffer_size",      "4194304",  # 4 MB UDP socket receive buffer
        "-fflags",           "nobuffer", # disable demuxer input buffer
        "-flags",            "low_delay",# decoder: no B-frame reorder buffer
        "-max_delay",        "0",        # RTP jitter buffer: 0 ms (default 500 ms)
        "-reorder_queue_size","0",       # RTP reorder queue off
        "-i", SDP,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-vf", f"scale={CAM_W}:{CAM_H}",
        "pipe:1",
    ]
    while not stop.is_set():
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            def _watch_stderr():
                for line in proc.stderr:
                    if b"error while decoding MB" in line:
                        corrupt.set()
            threading.Thread(target=_watch_stderr, daemon=True).start()

            print("[camera] FFmpeg started")
            while not stop.is_set():
                raw = proc.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                if corrupt.is_set():
                    corrupt.clear()  # discard corrupted frame, hold last good one
                else:
                    shared["frame"] = np.frombuffer(raw, dtype=np.uint8).reshape((CAM_H, CAM_W, 3)).copy()
            proc.terminate()
            proc.wait()
        except Exception as e:
            shared["cam_error"] = str(e)
        if not stop.is_set():
            time.sleep(1)


# ── helpers ────────────────────────────────────────────────────────────────────

def apply_deadband(v, db=DEADBAND):
    return 0.0 if abs(v) < db else v


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class DepthPID:
    _MAX_INTEGRAL = 500.0

    def __init__(self):
        self.integral   = 0.0
        self.prev_error = 0.0

    def reset(self, error: float = 0.0) -> None:
        self.integral   = 0.0
        self.prev_error = error

    def update(self, error: float, dt: float, kp: float, ki: float, kd: float) -> int:
        self.integral   = max(-self._MAX_INTEGRAL,
                              min(self._MAX_INTEGRAL, self.integral + error * dt))
        derivative      = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        output = kp * error + ki * self.integral + kd * derivative
        return max(0, min(1000, int(500 - output)))


def make_obs(frame_rgb, telem, depth_setpoint: float, pid_z: float):
    # 30-dim state, matching DAggerCollector.add_frame / controller_teleop.py's schema:
    # roll, pitch, yaw, depth, depth_setpoint, pid_z, acc_x/y/z, gyro_x/y/z,
    # baro_press_hpa, enclosure_temp_c, thr1..thr6_pwm, rollspeed/pitchspeed/yawspeed,
    # climb, vibration_x/y/z, ekf_compass/pos_horiz/pos_vert_variance. Requires a
    # checkpoint trained on this same 30-dim observation.state -- an older checkpoint
    # trained on a different schema will fail shape validation in the preprocessor.
    img_t   = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    state_t = torch.tensor([telem["roll"], telem["pitch"], telem["yaw"],
                             telem["depth"], depth_setpoint, pid_z,
                             telem["acc_x"],  telem["acc_y"],  telem["acc_z"],
                             telem["gyro_x"], telem["gyro_y"], telem["gyro_z"],
                             telem["baro_press_hpa"], telem["enclosure_temp_c"],
                             *telem["thr_pwm"],
                             telem["rollspeed"], telem["pitchspeed"], telem["yawspeed"],
                             telem["climb"],
                             telem["vibration_x"], telem["vibration_y"], telem["vibration_z"],
                             telem["ekf_compass_variance"], telem["ekf_pos_horiz_variance"],
                             telem["ekf_pos_vert_variance"]],
                            dtype=torch.float32)
    return {"observation.images.camera": img_t, "observation.state": state_t}


def action_to_commands(action):
    """Returns (x, y, dive, r, cam_pwm, gripper_open).
    dive is a depth-setpoint nudge (m per frame) matching the training action[2].
    The caller is responsible for running the PID to compute the actual z throttle."""
    x    = _clamp(int(action[0] * POLICY_GAIN * 1000), -1000, 1000)
    y    = _clamp(int(action[1] * POLICY_GAIN * 1000), -1000, 1000)
    dive = float(action[2])
    r    = _clamp(int(action[3] * POLICY_GAIN * 1000), -1000, 1000)
    cam  = _clamp(int(action[4] * 400 + 1500),          1100, 1900)
    gripper_open = bool(action[5] < 0.5)
    return x, y, dive, r, cam, gripper_open


# ── temporal ensembler ─────────────────────────────────────────────────────────

class TemporalEnsembler:
    def __init__(self, chunk_size, coeff):
        self.chunk_size = chunk_size
        self.coeff      = coeff
        self._lock   = threading.Lock()
        self._chunks: deque = deque(maxlen=20)
        self._step   = 0

    def add_chunk(self, chunk_np):
        with self._lock:
            self._chunks.append((chunk_np.copy(), self._step))

    def get_action(self):
        with self._lock:
            valid = [(c, s) for c, s in self._chunks
                     if 0 <= self._step - s < self.chunk_size]
            if not valid:
                if self._chunks:
                    c, s = self._chunks[-1]
                    idx = min(self._step - s, self.chunk_size - 1)
                    self._step += 1
                    return c[idx]
                return None
            actions = np.stack([c[self._step - s] for c, s in valid])
            ages    = np.array([self._step - s for _, s in valid], dtype=np.float32)
            weights = np.exp(-self.coeff * ages)
            weights /= weights.sum()
            self._step += 1
            return (weights[:, None] * actions).sum(axis=0)

    def reset(self):
        with self._lock:
            self._chunks.clear()
            self._step = 0


# ── policy worker ──────────────────────────────────────────────────────────────

class PolicyWorker:
    def __init__(self, policy, preprocessor, action_mean, action_std,
                 chunk_size, ensemble_coeff):
        self.policy       = policy
        self.preprocessor = preprocessor
        self._amean = action_mean
        self._astd  = action_std
        self._ensembler = TemporalEnsembler(chunk_size, ensemble_coeff)

        self._obs_lock = threading.Lock()
        self._frame          = None
        self._telem          = None
        self._depth_setpoint = 0.0
        self._pid_z          = 0.0

        self._running          = False
        self._thread           = None
        self.n_inferences      = 0
        self.last_inference_ms = 0.0
        self._inf_times: deque = deque(maxlen=6)  # monotonic timestamps of completions

    @property
    def avg_hz(self):
        times = list(self._inf_times)
        if len(times) < 2:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def update_obs(self, frame, telem, depth_setpoint: float, pid_z: float):
        with self._obs_lock:
            self._frame          = frame
            self._telem          = dict(telem)
            self._depth_setpoint = depth_setpoint
            self._pid_z          = pid_z

    def get_action(self):
        return self._ensembler.get_action()

    def start(self):
        self._running = True
        self._inf_times.clear()
        self._ensembler.reset()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            with self._obs_lock:
                frame          = self._frame
                telem          = dict(self._telem) if self._telem else None
                depth_setpoint = self._depth_setpoint
                pid_z          = self._pid_z

            if frame is None or telem is None:
                time.sleep(0.05)
                continue

            t0     = time.monotonic()
            obs    = make_obs(frame, telem, depth_setpoint, pid_z)
            obs_in = self.preprocessor(obs)

            with torch.no_grad():
                chunk_t = self.policy.predict_action_chunk(obs_in)

            chunk_np = chunk_t.squeeze(0).cpu().numpy()
            chunk_np = chunk_np * self._astd + self._amean

            self._ensembler.add_chunk(chunk_np)
            self.n_inferences      += 1
            self.last_inference_ms  = (time.monotonic() - t0) * 1000
            self._inf_times.append(time.monotonic())


# ── DAgger data collector ──────────────────────────────────────────────────────

class DAggerCollector:
    """LeRobot v3 dataset with an extra is_intervention bool column per frame.

    observation.state (30-dim, matching controller_teleop.py's schema): roll, pitch,
    yaw, depth, depth_setpoint, pid_z, acc_x/y/z, gyro_x/y/z (RAW_IMU), baro_press_hpa
    + enclosure_temp_c (SCALED_PRESSURE, internal enclosure), thr1..thr6_pwm
    (SERVO_OUTPUT_RAW, the 6 vectored thrusters' raw commanded PWM), rollspeed/
    pitchspeed/yawspeed (ATTITUDE, EKF-fused angular rates), climb (VFR_HUD, vertical
    velocity), vibration_x/y/z (VIBRATION), ekf_compass/pos_horiz/pos_vert_variance
    (EKF_STATUS_REPORT). SCALED_IMU2 (secondary onboard IMU) was considered but
    dropped -- this vehicle reports no second accel/gyro/mag (SYS_STATUS
    sensors_present bits unset, SCALED_IMU2 accel/gyro hard-zero), so it would only be
    dead weight. ekf_airspeed_variance/ekf_terrain_alt_variance/ekf_velocity_variance
    were dropped too -- confirmed hard-zero live via BlueOS: no airspeed sensor (N/A
    underwater, ArduSub doesn't support one), no rangefinder/altimeter (no
    DISTANCE_SENSOR ever received), no GPS or DVL for aided velocity
    (GPS_FIX_TYPE_NO_GPS, no VISION_POSITION_DELTA/ODOMETRY ever received).
    """

    CHUNKS_SIZE = 1000

    def __init__(self, task_name: str, root: Path) -> None:
        self.task_name = task_name
        self.root      = root
        (root / "meta").mkdir(parents=True, exist_ok=True)

        info_path = root / "meta" / "info.json"
        if info_path.exists():
            self.info             = json.loads(info_path.read_text())
            self.total_episodes   = self.info["total_episodes"]
            self.total_frames     = self.info["total_frames"]
            self.total_successes  = self.info.get("total_successes", 0)
        else:
            self.total_episodes   = 0
            self.total_frames     = 0
            self.total_successes  = 0
            self.info             = self._build_info()
            self._flush_info()
            self._write_tasks_parquet()

        self._recording    = False
        self._rows: list   = []
        self._container    = None
        self._stream       = None
        self._vpath: Path  = Path()
        self._ep_frame_idx = 0

    @property
    def recording(self) -> bool:
        return self._recording

    def start(self) -> bool:
        if self._recording:
            return False
        ep    = self.total_episodes
        chunk = ep // self.CHUNKS_SIZE
        vdir  = (self.root / "videos" / "observation.images.camera"
                 / f"chunk-{chunk:03d}")
        vdir.mkdir(parents=True, exist_ok=True)
        self._vpath     = vdir / f"episode_{ep:06d}.mp4"
        self._container = av.open(str(self._vpath), mode="w")
        self._stream    = self._container.add_stream("libx264", rate=SEND_HZ)
        self._stream.width   = CAM_W
        self._stream.height  = CAM_H
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"preset": "fast", "crf": "23"}

        self._rows         = []
        self._ep_frame_idx = 0
        self._recording    = True
        return True

    def add_frame(self, frame, telem: dict, x: int, y: int,
                  z: int, r: int, cam_tilt: int, gripper_closed: bool,
                  is_intervention: bool,
                  dive: float = 0.0, depth_setpoint: float = 0.0) -> None:
        if not self._recording:
            return
        img = (frame if frame is not None
               else np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8))
        av_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        av_frame.pts = self._ep_frame_idx
        for pkt in self._stream.encode(av_frame.reformat(format="yuv420p")):
            self._container.mux(pkt)

        self._rows.append({
            "obs": np.array([telem["roll"], telem["pitch"], telem["yaw"],
                             telem["depth"], depth_setpoint,
                             (z - 500) / 500.0,
                             telem["acc_x"],  telem["acc_y"],  telem["acc_z"],
                             telem["gyro_x"], telem["gyro_y"], telem["gyro_z"],
                             telem["baro_press_hpa"], telem["enclosure_temp_c"],
                             *telem["thr_pwm"],
                             telem["rollspeed"], telem["pitchspeed"], telem["yawspeed"],
                             telem["climb"],
                             telem["vibration_x"], telem["vibration_y"], telem["vibration_z"],
                             telem["ekf_compass_variance"], telem["ekf_pos_horiz_variance"],
                             telem["ekf_pos_vert_variance"]],
                            dtype=np.float32),
            "act": np.array([x / 1000.0, y / 1000.0,
                             dive, r / 1000.0,
                             (cam_tilt - 1500) / 400.0,
                             1.0 if gripper_closed else 0.0], dtype=np.float32),
            "ts":  self._ep_frame_idx / float(SEND_HZ),
            "fi":  self._ep_frame_idx,
            "ei":  self.total_episodes,
            "gi":  self.total_frames + self._ep_frame_idx,
            "ii":  is_intervention,
        })
        self._ep_frame_idx += 1

    def finish(self, success: bool):
        if not self._recording:
            return None
        self._recording = False

        for pkt in self._stream.encode():
            self._container.mux(pkt)
        self._container.close()

        if not self._rows:
            try: self._vpath.unlink()
            except FileNotFoundError: pass
            return None

        ep    = self.total_episodes
        chunk = ep // self.CHUNKS_SIZE
        ddir  = self.root / "data" / f"chunk-{chunk:03d}"
        ddir.mkdir(parents=True, exist_ok=True)
        self._write_parquet(ddir / f"episode_{ep:06d}.parquet")

        n = len(self._rows)
        dataset_from        = self.total_frames
        self.total_frames   += n
        self.total_episodes += 1
        if success:
            self.total_successes += 1

        self._append_episodes_parquet(ep, n, chunk, dataset_from, success)
        self._update_stats()
        self._flush_info()
        return ep, n, success

    def _build_info(self) -> dict:
        return {
            "codebase_version": "v3.0",
            "robot_type":       "bluerov2",
            "total_episodes":   0,
            "total_frames":     0,
            "total_tasks":      1,
            "chunks_size":      self.CHUNKS_SIZE,
            "fps":              float(SEND_HZ),
            "splits":           {"train": "0:0"},
            "data_path":  "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet",
            "video_path": ("videos/{video_key}/chunk-{chunk_index:03d}"
                           "/episode_{file_index:06d}.mp4"),
            "features": {
                "observation.images.camera": {
                    "dtype": "video",
                    "shape": [CAM_H, CAM_W, 3],
                    "names": ["height", "width", "channels"],
                    "video_info": {
                        "video.fps":          float(SEND_HZ),
                        "video.codec":        "h264",
                        "video.pix_fmt":      "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio":          False,
                    },
                },
                "observation.state": {
                    "dtype": "float32", "shape": [30],
                    "names": ["roll", "pitch", "yaw", "depth",
                              "depth_setpoint", "pid_z",
                              "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                              "baro_press_hpa", "enclosure_temp_c",
                              "thr1_pwm", "thr2_pwm", "thr3_pwm",
                              "thr4_pwm", "thr5_pwm", "thr6_pwm",
                              "rollspeed", "pitchspeed", "yawspeed", "climb",
                              "vibration_x", "vibration_y", "vibration_z",
                              "ekf_compass_variance",
                              "ekf_pos_horiz_variance", "ekf_pos_vert_variance"],
                },
                "action": {
                    "dtype": "float32", "shape": [6],
                    "names": ["x", "y", "dive", "r", "cam_tilt", "gripper"],
                },
                "timestamp":       {"dtype": "float32", "shape": [1], "names": None},
                "frame_index":     {"dtype": "int64",   "shape": [1], "names": None},
                "episode_index":   {"dtype": "int64",   "shape": [1], "names": None},
                "index":           {"dtype": "int64",   "shape": [1], "names": None},
                "task_index":      {"dtype": "int64",   "shape": [1], "names": None},
                "next.done":       {"dtype": "bool",    "shape": [1], "names": None},
                "is_intervention": {"dtype": "bool",    "shape": [1], "names": None},
            },
        }

    def _flush_info(self) -> None:
        self.info.update({
            "total_episodes":  self.total_episodes,
            "total_frames":    self.total_frames,
            "total_successes": self.total_successes,
            "splits":          {"train": f"0:{self.total_episodes}"},
        })
        (self.root / "meta" / "info.json").write_text(json.dumps(self.info, indent=2))

    def _write_tasks_parquet(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(
            pa.table({
                "task_index": pa.array([0], pa.int64()),
                "task":       pa.array([self.task_name], pa.string()),
            }),
            str(self.root / "meta" / "tasks.parquet"),
        )

    def _append_episodes_parquet(self, ep: int, n: int, chunk: int,
                                  dataset_from: int, success: bool) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        ep_path = self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        ep_path.parent.mkdir(parents=True, exist_ok=True)
        fps = float(SEND_HZ)
        new_row = pa.table({
            "episode_index":    pa.array([ep],                    pa.int64()),
            "tasks":            pa.array([[self.task_name]],      pa.list_(pa.string())),
            "length":           pa.array([n],                     pa.int64()),
            "success":          pa.array([success],               pa.bool_()),
            "data/chunk_index": pa.array([chunk],                 pa.int64()),
            "data/file_index":  pa.array([ep],                    pa.int64()),
            "dataset_from_index":         pa.array([dataset_from],     pa.int64()),
            "dataset_to_index":           pa.array([dataset_from + n], pa.int64()),
            "videos/observation.images.camera/chunk_index":     pa.array([chunk], pa.int64()),
            "videos/observation.images.camera/file_index":      pa.array([ep],    pa.int64()),
            "videos/observation.images.camera/from_timestamp":  pa.array([0.0],   pa.float64()),
            "videos/observation.images.camera/to_timestamp":    pa.array([n / fps], pa.float64()),
            "meta/episodes/chunk_index": pa.array([0], pa.int64()),
            "meta/episodes/file_index":  pa.array([0], pa.int64()),
        })
        if ep_path.exists():
            merged = pa.concat_tables([pq.read_table(str(ep_path)), new_row])
        else:
            merged = new_row
        pq.write_table(merged, str(ep_path))

    def _update_stats(self) -> None:
        import pyarrow.parquet as pq
        all_obs, all_act, all_ts = [], [], []
        for pf in sorted((self.root / "data").rglob("*.parquet")):
            t = pq.read_table(str(pf), columns=["observation.state", "action", "timestamp"])
            all_obs.append(np.array(t["observation.state"].to_pylist(), dtype=np.float32))
            all_act.append(np.array(t["action"].to_pylist(), dtype=np.float32))
            all_ts.append(np.array(t["timestamp"].to_pylist(), dtype=np.float32)[:, None])
        if not all_obs:
            return

        def feat_stats(arr: np.ndarray) -> dict:
            qs = np.quantile(arr, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
            return {
                "mean":  arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist(),
                "min":   arr.min(axis=0).tolist(),  "max": arr.max(axis=0).tolist(),
                "count": int(len(arr)),
                "q01": qs[0].tolist(), "q10": qs[1].tolist(), "q50": qs[2].tolist(),
                "q90": qs[3].tolist(), "q99": qs[4].tolist(),
            }

        stats = {
            "observation.state": feat_stats(np.vstack(all_obs)),
            "action":            feat_stats(np.vstack(all_act)),
            "timestamp":         feat_stats(np.vstack(all_ts)),
        }
        (self.root / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    def _write_parquet(self, path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        n    = len(self._rows)
        done = [False] * n
        done[-1] = True
        table = pa.table({
            "observation.state": pa.array([r["obs"].tolist() for r in self._rows],
                                          pa.list_(pa.float32(), 30)),
            "action":            pa.array([r["act"].tolist() for r in self._rows],
                                          pa.list_(pa.float32(), 6)),
            "timestamp":         pa.array([r["ts"] for r in self._rows], pa.float32()),
            "frame_index":       pa.array([r["fi"] for r in self._rows], pa.int64()),
            "episode_index":     pa.array([r["ei"] for r in self._rows], pa.int64()),
            "index":             pa.array([r["gi"] for r in self._rows], pa.int64()),
            "task_index":        pa.array([0] * n,                       pa.int64()),
            "next.done":         pa.array(done,                          pa.bool_()),
            "is_intervention":   pa.array([r["ii"] for r in self._rows], pa.bool_()),
        })
        pq.write_table(table, str(path))


# ── checkpoint ─────────────────────────────────────────────────────────────────

def resolve_checkpoint(path_str):
    p = Path(path_str).resolve()
    if p.suffix == ".zip":
        out_dir = p.parent / p.stem
        if not out_dir.exists():
            print(f"[deploy] Extracting {p.name}...")
            with zipfile.ZipFile(p) as zf:
                zf.extractall(out_dir)
        else:
            print(f"[deploy] Using cached extraction at {out_dir}/")
        hits = sorted(out_dir.rglob("config.json"))
        return hits[0].parent if hits else out_dir
    return p


def patch_checkpoint_config(checkpoint_path: Path) -> None:
    """Remove fields from config.json that don't exist in the installed ACTConfig.
    Older checkpoints may carry stale fields that cause draccus to raise DecodingError."""
    cfg_path = checkpoint_path / "config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    try:
        import dataclasses
        from lerobot.policies.act.modeling_act import ACTConfig
        valid = {f.name for f in dataclasses.fields(ACTConfig)} | {"type"}
        stale = {k for k in cfg if k not in valid}
    except Exception:
        stale = {"use_peft", "pretrained_path", "push_to_hub",
                 "repo_id", "private", "tags", "license"} & cfg.keys()
    if stale:
        for k in stale:
            del cfg[k]
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"[deploy] Patched config.json: removed stale fields {stale}")


# ── HUD ────────────────────────────────────────────────────────────────────────

def draw_hud(screen, font, telem, armed, mode, x, y, z, r, cam_tilt,
             depth_setpoint: float = 0.0, lights: int = LIGHT_MIN,
             worker=None, collector=None):
    hud = pygame.Surface((WIN_W, HUD_H))
    hud.fill((248, 248, 248))
    arm_col  = (0, 180, 0) if armed else (255, 0, 0)
    mode_col = {"IDLE":   (0,   0,   0),
                "POLICY": (0,   0, 255),
                "HUMAN":  (255, 0,   0),
                "TEST":   (128, 0, 255)}.get(mode, (0, 0, 0))

    depth_err = depth_setpoint - telem["depth"]

    hz_str = ""
    if worker and mode in ("POLICY", "HUMAN", "TEST"):
        hz_str = (f"  infer={worker.last_inference_ms:.0f}ms"
                  f"  {worker.avg_hz:.1f}Hz  n={worker.n_inferences}")

    ep_count  = collector.total_episodes  if collector else 0
    successes = collector.total_successes if collector else 0
    if collector and collector.recording:
        ep_str  = f"  ● REC  ep:{ep_count}  f:{collector._ep_frame_idx}  success:{successes}/{ep_count}"
        ep_col  = (255, 0, 0)
    else:
        ep_str  = f"  success:{successes}/{ep_count}"
        ep_col  = (0, 0, 0)

    lines = [
        (f"{'● ARMED' if armed else '○ DISARMED':12s}   "
         f"Roll:{telem['roll']:+.2f}  Pitch:{telem['pitch']:+.2f}  "
         f"Yaw:{telem['yaw']:+.2f}  Depth:{telem['depth']:.2f}m  "
         f"Batt:{telem['batt_v']:.1f}V {telem['batt_mah']}mAh", arm_col),
        (f"DepthSP:{depth_setpoint:.2f}m  err:{depth_err:+.2f}m  "
         f"thrtl:{z}  fwd:{x:+d}  lat:{y:+d}  yaw:{r:+d}  cam:{cam_tilt}  lights:{lights}",
         (0, 0, 255)),
        (f"Mode:{mode:6s}{hz_str}", mode_col),
        (f"{ep_str.strip()}   "
         "Start=arm  Back=disarm  A=ep  X=ok  B=fail  Y=test  RT=grip  LB/RB=depth  Dpad=cam",
         ep_col),
    ]
    alerts = compute_alerts(telem)
    if alerts:
        lines.append((f"⚠ {'  |  '.join(alerts)}", (220, 0, 0)))
    else:
        lines.append(("Power/sensors OK", (0, 140, 0)))
    for i, (text, col) in enumerate(lines):
        hud.blit(font.render(text, True, col), (12, 10 + i * 34))
    screen.blit(hud, (0, CAM_H))


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="BlueROV2 controller + ACT policy with DAgger data collection")
    ap.add_argument("--checkpoint", required=True,
                    help="Checkpoint dir or .zip")
    ap.add_argument("--task-name", default="dagger",
                    help="Task name written into the dataset (default: dagger)")
    ap.add_argument("--data-dir", default=None,
                    help="Root for DAgger data (default: <repo>/dagger_data/<task-name>/)")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="Steps to execute per predicted chunk (default: policy chunk_size)")
    ap.add_argument("--ensemble-coeff", type=float, default=None,
                    help="Temporal ensembling decay coeff (default: from policy config). "
                         "Lower = trust older chunks more (smoother). "
                         "Higher = weight recent chunks heavily (more reactive).")
    args = ap.parse_args()

    task_name = args.task_name
    data_root = (Path(args.data_dir) if args.data_dir
                 else DATA_ROOT / "dagger_data" / task_name)

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    patch_checkpoint_config(checkpoint_path)
    print(f"[deploy] checkpoint : {checkpoint_path}")
    print(f"[deploy] dagger data: {data_root}")

    try:
        r = _session.get(
            f"{MAV}/mavlink/vehicles/1/components/1/messages/HEARTBEAT", timeout=3)
        assert r.ok
        print("[deploy] Connected to BlueROV2")
    except Exception:
        sys.exit(f"[deploy] Cannot reach {MAV} -- is the BlueROV2 connected?")

    print("[deploy] Loading policy...")
    from lerobot.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy.from_pretrained(str(checkpoint_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.eval().to(device)
    print(f"[deploy] Policy on {device}  chunk_size={policy.config.chunk_size}")

    from lerobot.policies.factory import make_pre_post_processors
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print("[deploy] Preprocessor loaded from checkpoint")

    from safetensors import safe_open
    sfp = checkpoint_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    with safe_open(str(sfp), framework="pt") as f:
        action_mean = f.get_tensor("action.mean").numpy()
        action_std  = f.get_tensor("action.std").numpy()
    print(f"[deploy] Action mean={action_mean.round(3)}  std={action_std.round(3)}")

    coeff        = args.ensemble_coeff or policy.config.temporal_ensemble_coeff or 0.01
    n_act_steps  = args.n_action_steps or policy.config.chunk_size
    worker = PolicyWorker(
        policy, preprocessor, action_mean, action_std,
        chunk_size=n_act_steps,
        ensemble_coeff=coeff,
    )
    print(f"[deploy] TemporalEnsembler coeff={coeff}  "
          f"chunk_size={policy.config.chunk_size}  n_action_steps={n_act_steps}")

    collector = DAggerCollector(task_name, data_root)
    print(f"[deploy] DAgger collector ready  (saved episodes: {collector.total_episodes})")

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        sys.exit("No Xbox controller found -- plug it in and retry.")
    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"[deploy] Controller: {joy.get_name()}")

    telem = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "depth": 0.0,
             "batt_v": 0.0, "batt_mah": 0,
             "acc_x": 0.0, "acc_y": 0.0, "acc_z": 0.0,
             "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0,
             "baro_press_hpa": 0.0, "enclosure_temp_c": 0.0, "thr_pwm": [1500] * 6,
             "vcc_mv": 0, "power_flags": 0,
             "sensors_enabled": 0, "sensors_health": 0, "sensors_present": 0,
             "rollspeed": 0.0, "pitchspeed": 0.0, "yawspeed": 0.0, "climb": 0.0,
             "vibration_x": 0.0, "vibration_y": 0.0, "vibration_z": 0.0,
             "ekf_compass_variance": 0.0,
             "ekf_pos_horiz_variance": 0.0, "ekf_pos_vert_variance": 0.0}
    cam   = {"frame": None, "cam_error": None}
    stop  = threading.Event()
    for target, targs in [
        (telemetry_loop, (telem, stop)),
        (camera_loop,    (cam,   stop)),
        (_sender_loop,   (stop,)),
    ]:
        threading.Thread(target=target, args=targs, daemon=True).start()

    _set_servo_function(11, 0.0)
    _set_servo_function(13, 0.0)
    _set_servo_function(14, 0.0)
    _set_servo_function(16, 0.0)
    for m in (1, 2, 5):
        _set_motor_direction(m, -1)
    _set_message_interval(RAW_IMU_MSG_ID, RAW_IMU_HZ)
    print(f"[deploy] Servos claimed, motor directions set (1, 2, 5 reversed), "
          f"RAW_IMU requested @ {RAW_IMU_HZ}Hz")

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("BlueROV2 DAgger")
    font  = pygame.font.SysFont("monospace", 13)
    clock = pygame.time.Clock()
    placeholder = pygame.Surface((CAM_W, CAM_H))
    placeholder.fill((30, 30, 30))
    placeholder.blit(font.render("Waiting for camera...", True, (120, 120, 120)),
                     (CAM_W // 2 - 80, CAM_H // 2))

    # ── state ─────────────────────────────────────────────────────────────────
    armed             = False
    mode              = "IDLE"   # IDLE | POLICY | HUMAN | TEST
    cam_tilt          = 1750
    prev_cam_tilt     = -1
    prev_rt_pressed   = False
    prev_gripper_open = True
    prev_cam_pwm      = cam_tilt
    last_hb           = 0.0
    prev_btns         = {}
    depth_setpoint    = 0.0
    depth_pid         = DepthPID()
    lights            = LIGHT_MIN
    prev_lights       = -1
    x = y = r = 0
    z = 500
    prev_alerts: list = []

    def just_pressed(btn):
        return joy.get_button(btn) and not prev_btns.get(btn, False)

    print("[deploy] Ready -- Start=arm  A=start-episode  X=success  B=fail  Back=disarm")

    try:
        while True:
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt

            # ── read axes ────────────────────────────────────────────────────
            fwd  = apply_deadband(-joy.get_axis(AX_LEFT_Y))
            lat  = apply_deadband( joy.get_axis(AX_LEFT_X))
            yaw  = apply_deadband( joy.get_axis(AX_RIGHT_X))
            rt_pressed = joy.get_axis(AX_RT) > TRIGGER_THRESHOLD

            # ── button edges ─────────────────────────────────────────────────
            if just_pressed(BTN_START):
                if set_arm(True):
                    armed = True
                    depth_setpoint = telem["depth"]
                    depth_pid.reset()
                    print(f"[deploy] Armed  (depth setpoint → {depth_setpoint:.2f} m)")

            if just_pressed(BTN_BACK):
                if set_arm(False):
                    armed = False
                    print("[deploy] Disarmed")

            if just_pressed(BTN_A) and mode == "IDLE":
                if collector.start():
                    worker.start()
                    mode = "POLICY"
                    print(f"[deploy] Episode {collector.total_episodes} started -> POLICY")

            if just_pressed(BTN_X) and mode in ("POLICY", "HUMAN"):
                result = collector.finish(success=True)
                worker.stop()
                mode = "IDLE"
                if result:
                    ep, n, _ = result
                    print(f"[deploy] Episode {ep} saved SUCCESS ({n} frames)")

            if just_pressed(BTN_B) and mode in ("POLICY", "HUMAN"):
                result = collector.finish(success=False)
                worker.stop()
                mode = "IDLE"
                if result:
                    ep, n, _ = result
                    print(f"[deploy] Episode {ep} saved FAILURE ({n} frames)")

            if just_pressed(BTN_Y):
                if mode == "IDLE":
                    worker.start()
                    mode = "TEST"
                    print("[deploy] TEST mode: policy running, no recording")
                elif mode == "TEST":
                    worker.stop()
                    mode = "IDLE"
                    print("[deploy] TEST mode off")

            prev_btns = {b: joy.get_button(b)
                         for b in [BTN_A, BTN_B, BTN_X, BTN_Y,
                                   BTN_LB, BTN_RB, BTN_START, BTN_BACK]}

            # ── bumpers: depth setpoint nudge (all modes, matches training) ──
            human_dive = 0.0
            depth_step = DEPTH_RATE / SEND_HZ * DEPTH_GAIN
            if joy.get_button(BTN_RB):
                human_dive = -depth_step
                depth_setpoint -= depth_step
            if joy.get_button(BTN_LB):
                human_dive = -depth_step
                depth_setpoint += depth_step
            depth_setpoint = max(DEPTH_MIN, min(DEPTH_MAX, depth_setpoint))

            # ── d-pad: camera tilt (up/down), lights (left/right) — hold to change
            hat = joy.get_hat(0) if joy.get_numhats() > 0 else (0, 0)
            if hat[1] == 1:
                cam_tilt = max(1100, cam_tilt - CAM_TILT_STEP)
            elif hat[1] == -1:
                cam_tilt = min(1900, cam_tilt + CAM_TILT_STEP)
            if hat[0] == -1:
                lights = max(LIGHT_MIN, lights - LIGHT_STEP)
            elif hat[0] == 1:
                lights = min(LIGHT_MAX, lights + LIGHT_STEP)

            # ── gripper (always active) ───────────────────────────────────────
            if rt_pressed and not prev_rt_pressed:
                set_gripper(False)
                print("[deploy] Gripper: close")
            elif not rt_pressed and prev_rt_pressed:
                set_gripper(True)
                print("[deploy] Gripper: open")
            prev_rt_pressed = rt_pressed

            # ── PID z (used by all modes) ─────────────────────────────────────
            depth_error = depth_setpoint - telem["depth"]
            z = depth_pid.update(depth_error, 1.0 / SEND_HZ,
                                 KP_DEPTH, KI_DEPTH, KD_DEPTH)
            pid_z_norm  = (z - 500) / 500.0

            # ── intervention detection ────────────────────────────────────────
            is_intervening = (max(abs(fwd), abs(lat), abs(yaw)) > INTERVENTION_THRESH)

            # ── mode transitions (POLICY <-> HUMAN during recording) ──────────
            if mode == "POLICY" and is_intervening:
                mode = "HUMAN"
                cam_tilt = prev_cam_pwm  # sync manual cam_tilt to last policy value
            elif mode == "HUMAN" and (just_pressed(BTN_A) or just_pressed(BTN_Y)):
                mode = "POLICY"
                print("[deploy] Policy re-enabled by user")

            # ── compute and send commands ─────────────────────────────────────
            gripper_closed = rt_pressed

            if mode == "IDLE":
                x = int(fwd * 1000 * GAIN)
                y = int(lat * 1000 * GAIN)
                r = int(yaw * 1000 * GAIN * ROT_GAIN)
                if armed:
                    send_manual_control(x, y, z, r)
                if cam_tilt != prev_cam_tilt:
                    set_camera_tilt(cam_tilt)
                    prev_cam_tilt = cam_tilt

            elif mode == "HUMAN":
                x = int(fwd * 1000 * GAIN)
                y = int(lat * 1000 * GAIN)
                r = int(yaw * 1000 * GAIN * ROT_GAIN)
                if armed:
                    send_manual_control(x, y, z, r)
                if cam_tilt != prev_cam_tilt:
                    set_camera_tilt(cam_tilt)
                    prev_cam_tilt = cam_tilt
                frame = cam["frame"]
                if frame is not None:
                    worker.update_obs(frame, telem, depth_setpoint, pid_z_norm)
                collector.add_frame(cam["frame"], telem, x, y, z, r,
                                    cam_tilt, gripper_closed,
                                    is_intervention=True,
                                    dive=human_dive,
                                    depth_setpoint=depth_setpoint)

            elif mode == "TEST":
                frame = cam["frame"]
                if frame is not None:
                    worker.update_obs(frame, telem, depth_setpoint, pid_z_norm)
                action = worker.get_action()
                if action is not None:
                    x, y, policy_dive, r, cam_pwm, grip_open = action_to_commands(action)
                    depth_setpoint = max(DEPTH_MIN, min(DEPTH_MAX,
                                                        depth_setpoint + policy_dive))
                    if armed:
                        send_manual_control(x, y, z, r)
                    if cam_pwm != prev_cam_pwm:
                        set_camera_tilt(cam_pwm)
                        prev_cam_pwm = cam_pwm
                    if grip_open != prev_gripper_open:
                        set_gripper(grip_open)
                        prev_gripper_open = grip_open
                else:
                    if armed:
                        send_manual_control(0, 0, z, 0)

            else:  # POLICY
                frame = cam["frame"]
                if frame is not None:
                    worker.update_obs(frame, telem, depth_setpoint, pid_z_norm)
                action = worker.get_action()
                if action is not None:
                    x, y, policy_dive, r, cam_pwm, grip_open = action_to_commands(action)
                    depth_setpoint = max(DEPTH_MIN, min(DEPTH_MAX,
                                                        depth_setpoint + policy_dive))
                    if armed:
                        send_manual_control(x, y, z, r)
                    if cam_pwm != prev_cam_pwm:
                        set_camera_tilt(cam_pwm)
                        prev_cam_pwm = cam_pwm
                    if grip_open != prev_gripper_open:
                        set_gripper(grip_open)
                        prev_gripper_open = grip_open
                    collector.add_frame(cam["frame"], telem, x, y, z, r,
                                        cam_pwm, not grip_open,
                                        is_intervention=False,
                                        dive=policy_dive,
                                        depth_setpoint=depth_setpoint)
                else:
                    # No action yet from ensembler; hover
                    if armed:
                        send_manual_control(0, 0, z, 0)
                    collector.add_frame(cam["frame"], telem, 0, 0, z, 0,
                                        cam_tilt, False,
                                        is_intervention=False,
                                        dive=0.0,
                                        depth_setpoint=depth_setpoint)

            # ── lights (always active) ───────────────────────────────────────
            if lights != prev_lights:
                set_lights(lights)
                prev_lights = lights

            # ── heartbeat ────────────────────────────────────────────────────
            now = time.monotonic()
            if now - last_hb >= 1.0:
                send_heartbeat()
                last_hb = now

            # ── health / power alerts ────────────────────────────────────────
            alerts = compute_alerts(telem)
            if alerts and alerts != prev_alerts:
                print(f"[ALERT] {'  |  '.join(alerts)}")
            prev_alerts = alerts

            # ── render ───────────────────────────────────────────────────────
            raw_frame = cam["frame"]
            if raw_frame is not None:
                screen.blit(
                    pygame.surfarray.make_surface(raw_frame.transpose(1, 0, 2)), (0, 0))
            else:
                screen.blit(placeholder, (0, 0))

            hud_cam = prev_cam_pwm if mode in ("POLICY", "TEST") else cam_tilt
            draw_hud(screen, font, telem, armed, mode, x, y, z, r, hud_cam,
                     depth_setpoint=depth_setpoint, lights=lights,
                     worker=worker if mode in ("POLICY", "HUMAN", "TEST") else None,
                     collector=collector)
            pygame.display.flip()
            clock.tick(SEND_HZ)

    except KeyboardInterrupt:
        pass
    finally:
        if mode in ("POLICY", "HUMAN"):
            collector.finish(success=False)
            worker.stop()
        elif mode == "TEST":
            worker.stop()
        stop.set()
        pygame.quit()
        if armed:
            set_arm(False)
        send_manual_control(0, 0, 500, 0)
        _set_servo_function(11, 184.0)
        _set_servo_function(13, 0.0)
        _set_servo_function(14, 0.0)
        _set_servo_function(16, 7.0)
        print("[deploy] Exited cleanly.")


if __name__ == "__main__":
    main()
