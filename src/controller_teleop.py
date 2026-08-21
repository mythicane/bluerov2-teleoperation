"""
Xbox controller teleop for BlueROV2.

Control layout:
  Left  stick Y    forward / back
  Left  stick X    strafe left / right
  Right stick X    yaw
  Right stick Y    ascend / descend
  RT (axis 5)      gripper close (hold) / open (release)
  D-pad up         camera tilt up
  D-pad down       camera tilt down
  D-pad left       lights dim
  D-pad right      lights brighten
  LB (button 4)    depth setpoint down (deeper)
  RB (button 5)    depth setpoint up (shallower)
  Start (button 7) arm
  Back  (button 6) disarm

Run from project root:
  python src/controller_teleop.py
"""

import sys
import time
import queue
import threading
import argparse
import json
import subprocess
from pathlib import Path

import av
import numpy as np
import requests
import pygame
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BLUEROV_URL, DATA_ROOT

# ── config ─────────────────────────────────────────────────────────────────────

SEND_HZ           = 50
GAIN              = 0.15    # scale joystick → lateral/yaw command
DEPTH_GAIN        = 0.5 # wrt to the main gain
ROT_GAIN          = 1.0
DEADBAND          = 0.005
TRIGGER_THRESHOLD = 0.5    # trigger axis value to register as a press
CAM_TILT_STEP     = 15     # PWM per tick while d-pad held
LIGHT_STEP        = 15     # PWM per tick while d-pad held
LIGHT_MIN         = 1100
LIGHT_MAX         = 1900

# depth PID controller
DEPTH_RATE = 0.5    # m/s setpoint adjustment rate at full stick deflection
KP_DEPTH   = 200.0  # proportional gain: error (m) → MANUAL_CONTROL z offset
KI_DEPTH   = 0.0    # integral gain
KD_DEPTH   = 0.0    # derivative gain
DEPTH_MIN  = -0.5   # m  (slack for surface bobbing)
DEPTH_MAX  = 50.0   # m
ENABLE_CAMERA = True   # camera_loop/ffmpeg on -- in-script video feed + window panel.
                       # Flip to False to disable if it conflicts with QGroundControl's feed.
CAM_W, CAM_H = 854, 480
HUD_H         = 230
WIN_W, WIN_H  = (CAM_W, CAM_H + HUD_H) if ENABLE_CAMERA else (CAM_W, HUD_H)

# Xbox Series X axis indices (pygame / XInput on Windows)
AX_LEFT_X  = 0   # strafe
AX_LEFT_Y  = 1   # forward/back  (negative = forward)
AX_RIGHT_X = 2   # yaw
AX_RIGHT_Y = 3   # ascend/descend (negative = up)
AX_LT      = 4   # gripper open  (-1 unpressed → +1 full)
AX_RT      = 5   # gripper close (-1 unpressed → +1 full)

# Xbox button indices
BTN_A     = 0
BTN_B     = 1
BTN_X     = 2
BTN_Y     = 3
BTN_LB    = 4
BTN_RB    = 5
BTN_BACK  = 6
BTN_START = 7

MAV    = f"{BLUEROV_URL}/mavlink2rest"
HEADER = {"system_id": 255, "component_id": 0, "sequence": 0}
SDP    = str(Path(__file__).parent.parent / "stream.sdp")

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


def _sensor_alert_text(telem: dict) -> str:
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


def compute_alerts(telem: dict) -> list:
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

# ── MAVLink2Rest ───────────────────────────────────────────────────────────────

_session     = requests.Session()
_mc_queue:    queue.Queue = queue.Queue(maxsize=1)   # MANUAL_CONTROL — drop stale
_servo_queue: queue.Queue = queue.Queue(maxsize=8)   # DO_SET_SERVO   — never drop


def _enqueue_mc(message: dict) -> None:
    """Overwrite any pending MANUAL_CONTROL with the latest values."""
    try:
        _mc_queue.put_nowait(message)
    except queue.Full:
        try:
            _mc_queue.get_nowait()
        except queue.Empty:
            pass
        _mc_queue.put_nowait(message)


def _enqueue_servo(message: dict) -> None:
    """Queue a servo command; drop only if the servo queue itself is full."""
    try:
        _servo_queue.put_nowait(message)
    except queue.Full:
        pass


def _sender_loop(stop: threading.Event) -> None:
    while not stop.is_set():
        # drain all pending servo commands first so they are never starved by MC
        while True:
            try:
                msg = _servo_queue.get_nowait()
                _session.post(f"{MAV}/mavlink",
                              json={"header": HEADER, "message": msg}, timeout=0.1)
            except queue.Empty:
                break
            except Exception:
                break
        # then send the latest manual-control if one is waiting
        try:
            msg = _mc_queue.get(timeout=0.02)
            _session.post(f"{MAV}/mavlink",
                          json={"header": HEADER, "message": msg}, timeout=0.1)
        except (queue.Empty, Exception):
            pass


def _post_sync(message: dict) -> bool:
    try:
        r = _session.post(f"{MAV}/mavlink",
                          json={"header": HEADER, "message": message}, timeout=0.5)
        return r.ok
    except Exception:
        return False


def _param_id_chars(name: str) -> list:
    return list(name.ljust(16, "\x00")[:16])


def _set_servo_function(servo: int, value: float) -> None:
    _post_sync({"type": "PARAM_SET", "target_system": 1, "target_component": 1,
                "param_id": _param_id_chars(f"SERVO{servo}_FUNCTION"),
                "param_value": value,
                "param_type": {"type": "MAV_PARAM_TYPE_INT8"}})


def _set_motor_direction(motor: int, direction: int) -> None:
    """Set MOT_n_DIRECTION: 1 = normal, -1 = reversed."""
    _post_sync({"type": "PARAM_SET", "target_system": 1, "target_component": 1,
                "param_id": _param_id_chars(f"MOT_{motor}_DIRECTION"),
                "param_value": float(direction),
                "param_type": {"type": "MAV_PARAM_TYPE_INT8"}})


def _set_message_interval(msg_id: int, hz: float) -> None:
    """MAV_CMD_SET_MESSAGE_INTERVAL: request the autopilot stream msg_id at hz."""
    _post_sync({"type": "COMMAND_LONG", "target_system": 1, "target_component": 1,
                "command": {"type": "MAV_CMD_SET_MESSAGE_INTERVAL"}, "confirmation": 0,
                "param1": float(msg_id), "param2": float(1_000_000 / hz),
                "param3": 0.0, "param4": 0.0, "param5": 0.0, "param6": 0.0, "param7": 0.0})


def _do_set_servo(channel: int, pwm: float) -> None:
    _enqueue_servo({"type": "COMMAND_LONG", "target_system": 1, "target_component": 1,
                    "command": {"type": "MAV_CMD_DO_SET_SERVO"}, "confirmation": 0,
                    "param1": float(channel), "param2": pwm,
                    "param3": 0.0, "param4": 0.0, "param5": 0.0, "param6": 0.0, "param7": 0.0})


def send_heartbeat() -> None:
    _post_sync({"type": "HEARTBEAT", "custom_mode": 0,
                "mavtype": {"type": "MAV_TYPE_GCS"},
                "autopilot": {"type": "MAV_AUTOPILOT_INVALID"},
                "base_mode": {"bits": 0},
                "system_status": {"type": "MAV_STATE_ACTIVE"},
                "mavlink_version": 3})


def set_arm(armed: bool) -> bool:
    return _post_sync({"type": "COMMAND_LONG", "target_system": 1, "target_component": 1,
                       "command": {"type": "MAV_CMD_COMPONENT_ARM_DISARM"},
                       "confirmation": 0,
                       "param1": 1.0 if armed else 0.0, "param2": 21196.0,
                       "param3": 0.0, "param4": 0.0,
                       "param5": 0.0, "param6": 0.0, "param7": 0.0})


def send_manual_control(x: int, y: int, z: int, r: int) -> None:
    _enqueue_mc({"type": "MANUAL_CONTROL", "target": 1,
                 "x": x, "y": y, "z": z, "r": r, "buttons": 0})


def set_gripper(open: bool) -> None:
    _do_set_servo(11, 1900.0 if open else 1100.0)


def set_camera_tilt(pwm: int) -> None:
    _do_set_servo(16, float(pwm))


def set_lights(pwm: int) -> None:
    _do_set_servo(13, float(pwm))
    _do_set_servo(14, float(pwm))


# ── background threads ─────────────────────────────────────────────────────────

def telemetry_loop(data: dict, stop: threading.Event) -> None:
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
        except Exception:
            pass
        time.sleep(0.1)


def config_watcher_loop(live_cfg: dict, stop: threading.Event) -> None:
    last_mtime = 0.0
    while not stop.is_set():
        try:
            mtime = _CONFIG_PATH.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                new_cfg = _load_pid_config()
                live_cfg.update(new_cfg)
                print(f"[config] reloaded  kp={new_cfg['kp_depth']}  "
                      f"ki={new_cfg['ki_depth']}  kd={new_cfg['kd_depth']}  "
                      f"rate={new_cfg['depth_rate']}  gain={new_cfg['gain']}  "
                      f"deadband={new_cfg['deadband']}")
        except Exception:
            pass
        time.sleep(1.0)


FFMPEG = "ffmpeg"


def camera_loop(shared: dict, stop: threading.Event) -> None:
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

            print("Camera: FFmpeg started")
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
def apply_deadband(value: float, db: float = DEADBAND) -> float:
    return 0.0 if abs(value) < db else value


# ── data collection ────────────────────────────────────────────────────────────

class DataCollector:
    """Writes a LeRobot v3 dataset (parquet + H264 MP4) to disk at SEND_HZ.

    Layout:
      <root>/
        meta/info.json  tasks.parquet  episodes/chunk-000/file-000.parquet  stats.json
        data/chunk-000/episode_XXXXXX.parquet
        videos/observation.images.camera/chunk-000/episode_XXXXXX.mp4

    Actions are normalised to [-1, 1]:  x/1000, y/1000, (z-500)/500, r/1000.
    State is [roll, pitch, yaw, depth, depth_setpoint, pid_z, acc_x, acc_y, acc_z,
    gyro_x, gyro_y, gyro_z, baro_press_hpa, enclosure_temp_c, thr1..thr6_pwm,
    rollspeed, pitchspeed, yawspeed, climb, vibration_x, vibration_y, vibration_z,
    ekf_compass_variance, ekf_pos_horiz_variance, ekf_pos_vert_variance].
    acc_*/gyro_* are the onboard IMU's raw body-frame linear acceleration (m/s^2) and
    angular velocity (rad/s) from RAW_IMU -- distinct from roll/pitch/yaw, which are
    EKF-fused attitude estimates. baro_press_hpa/enclosure_temp_c are the internal
    enclosure barometer (SCALED_PRESSURE, not the external depth sensor). thr*_pwm are
    the 6 vectored thrusters' raw commanded PWM (SERVO_OUTPUT_RAW, neutral ~1500).
    rollspeed/pitchspeed/yawspeed are EKF-fused angular rates (ATTITUDE), distinct from
    the raw gyro_*. climb is vertical velocity (VFR_HUD). vibration_* and the
    ekf_*_variance fields are hardware/estimator health signals (VIBRATION,
    EKF_STATUS_REPORT). SCALED_IMU2 (secondary onboard IMU) was considered but
    dropped -- this vehicle reports no second accel/gyro/mag. ekf_airspeed_variance/
    ekf_terrain_alt_variance/ekf_velocity_variance were dropped too -- confirmed
    hard-zero live via BlueOS: no airspeed sensor (N/A underwater, ArduSub doesn't
    support one), no rangefinder/altimeter (no DISTANCE_SENSOR ever received), no GPS
    or DVL for aided velocity (GPS_FIX_TYPE_NO_GPS, no VISION_POSITION_DELTA/ODOMETRY
    ever received). Timestamps are frame_index / fps (grid-aligned to the constant-fps
    video).
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

    # ── public ─────────────────────────────────────────────────────────────────

    @property
    def recording(self) -> bool:
        return self._recording

    def start(self) -> bool:
        """Open a new episode.  Returns False if already recording."""
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
                  dive: float = 0.0, depth_setpoint: float = 0.0) -> None:
        """Encode one video frame and append one data row. No-op if not recording."""
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
        })
        self._ep_frame_idx += 1

    def finish(self, success: bool):
        """Close the episode.  Returns (ep_index, n_frames, success) or None."""
        if not self._recording:
            return None
        self._recording = False

        for pkt in self._stream.encode():       # flush encoder
            self._container.mux(pkt)
        self._container.close()

        if not self._rows:                      # nothing recorded → discard
            try:
                self._vpath.unlink()
            except FileNotFoundError:
                pass
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

    # ── private ────────────────────────────────────────────────────────────────

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
                    "names": ["roll", "pitch", "yaw", "depth", "depth_setpoint", "pid_z",
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
                "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
                "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
                "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
                "index":         {"dtype": "int64",   "shape": [1], "names": None},
                "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
                "next.done":     {"dtype": "bool",    "shape": [1], "names": None},
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
        })
        pq.write_table(table, str(path))


# ── HUD ───────────────────────────────────────────────────────────────────────

def draw_hud(screen, font, telem, armed, x, y, z, r, cam_tilt,
             depth_setpoint: float = 0.0, lights: int = LIGHT_MIN,
             recording: bool = False, ep_count: int = 0, successes: int = 0):
    hud = pygame.Surface((WIN_W, HUD_H))
    hud.fill((248, 248, 248))
    arm_col = (0, 180, 0)    if armed     else (255, 0, 0)
    rec_col = (255, 0, 0)    if recording else (0, 0, 0)
    rec_str = f"● REC  ep:{ep_count}" if recording else "○ IDLE"
    depth_err = depth_setpoint - telem["depth"]
    lines = [
        (f"{'● ARMED' if armed else '○ DISARMED':12s}   "
         f"Roll:{telem['roll']:+.2f}  Pitch:{telem['pitch']:+.2f}  "
         f"Yaw:{telem['yaw']:+.2f}  Depth:{telem['depth']:.2f}m  "
         f"Batt:{telem['batt_v']:.1f}V {telem['batt_mah']}mAh", arm_col),
        (f"DepthSP:{depth_setpoint:.2f}m  err:{depth_err:+.2f}m  "
         f"thrtl:{z}  fwd:{x:+d}  lat:{y:+d}  yaw:{r:+d}  cam:{cam_tilt}  lights:{lights}",
         (0, 0, 255)),
        (f"IMU acc(m/s2):{telem['acc_x']:+.2f},{telem['acc_y']:+.2f},{telem['acc_z']:+.2f}  "
         f"gyro(rad/s):{telem['gyro_x']:+.3f},{telem['gyro_y']:+.3f},{telem['gyro_z']:+.3f}",
         (100, 0, 150)),
        (f"{rec_str}   success:{successes}/{ep_count}", rec_col),
    ]
    alerts = compute_alerts(telem)
    if alerts:
        lines.append((f"⚠ {'  |  '.join(alerts)}", (220, 0, 0)))
    else:
        lines.append(("Power/sensors OK", (0, 140, 0)))
    lines.append(
        ("Start=arm  Back=disarm  A=rec  X=success  B=fail  RT=grip  LB/RB=depth  Dpad=cam/lights",
         (0, 0, 0)))
    for i, (text, col) in enumerate(lines):
        hud.blit(font.render(text, True, col), (12, 10 + i * 34))
    screen.blit(hud, (0, CAM_H if ENABLE_CAMERA else 0))


# ── depth PID controller ───────────────────────────────────────────────────────

class DepthPID:
    _MAX_INTEGRAL = 1.0  # clamp to prevent windup

    def __init__(self):
        self.integral   = 0.0
        self.prev_error = 0.0

    def reset(self, error: float = 0.0) -> None:
        self.integral   = 0.0
        self.prev_error = error

    def update(self, error: float, dt: float, kp: float, ki: float, kd: float) -> int:
        self.integral   = max(-self._MAX_INTEGRAL, min(self._MAX_INTEGRAL, self.integral + error * dt))
        derivative      = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        output = kp * error + ki * self.integral + kd * derivative
        return max(0, min(1000, int(500 - output)))


# ── config loader ──────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "pid_configs.yaml"


def _load_pid_config() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}

    def get(section, key, default):
        return cfg.get(section, {}).get(key, default)

    return {
        "kp_depth":   get("depth",   "kp",         KP_DEPTH),
        "ki_depth":   get("depth",   "ki",         KI_DEPTH),
        "kd_depth":   get("depth",   "kd",         KD_DEPTH),
        "depth_rate": get("depth",   "depth_rate",  DEPTH_RATE),
        "depth_min":  get("depth",   "depth_min",   DEPTH_MIN),
        "depth_max":  get("depth",   "depth_max",   DEPTH_MAX),
        "gain":       get("lateral", "gain",         GAIN),
        "deadband":   get("lateral", "deadband",     DEADBAND),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main(task_name: str) -> None:
    live_cfg = _load_pid_config()
    print(f"[config] kp={live_cfg['kp_depth']}  rate={live_cfg['depth_rate']}  "
          f"gain={live_cfg['gain']}  deadband={live_cfg['deadband']}  "
          f"depth=[{live_cfg['depth_min']}, {live_cfg['depth_max']}]")

    # connectivity check
    try:
        r = _session.get(
            f"{MAV}/mavlink/vehicles/1/components/1/messages/HEARTBEAT", timeout=3)
        assert r.ok
        print("Connected to BlueROV2")
    except Exception:
        print(f"Cannot reach {MAV}")
        return

    # controller init
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No controller found — plug in the Xbox controller and retry.")
        pygame.quit()
        return
    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"Controller: {joy.get_name()}  "
          f"(axes={joy.get_numaxes()}  buttons={joy.get_numbuttons()})")

    # take servo ownership (set to passthrough so DO_SET_SERVO works)
    _set_servo_function(11, 0.0)
    _set_servo_function(13, 0.0)
    _set_servo_function(14, 0.0)
    _set_servo_function(16, 0.0)
    print("Servos ready")

    # motor direction — motors 1, 2, 5 are physically reversed on this vehicle
    for m in (1, 2, 5):
        _set_motor_direction(m, -1)
    print("Motor directions set (1, 2, 5 reversed)")

    _set_message_interval(RAW_IMU_MSG_ID, RAW_IMU_HZ)
    print(f"Requested RAW_IMU @ {RAW_IMU_HZ}Hz")

    data_root = DATA_ROOT / task_name
    collector = DataCollector(task_name, data_root)
    print(f"Data collector ready → {data_root}  ({collector.total_episodes} episodes already saved)")

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
    threads = [
        threading.Thread(target=telemetry_loop,    args=(telem,    stop), daemon=True),
        threading.Thread(target=_sender_loop,      args=(stop,),          daemon=True),
        threading.Thread(target=config_watcher_loop, args=(live_cfg, stop), daemon=True),
    ]
    if ENABLE_CAMERA:
        threads.append(threading.Thread(target=camera_loop, args=(cam, stop), daemon=True))
    for t in threads:
        t.start()

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("BlueROV2 Controller Teleop")
    font  = pygame.font.SysFont("monospace", 13)
    clock = pygame.time.Clock()

    placeholder = pygame.Surface((CAM_W, CAM_H))
    placeholder.fill((30, 30, 30))
    placeholder.blit(font.render("Waiting for camera…", True, (120, 120, 120)),
                     (CAM_W // 2 - 80, CAM_H // 2))

    armed           = False
    depth_setpoint  = 0.0  # metres; reset to actual depth on arm
    depth_pid       = DepthPID()
    cam_tilt        = 1750
    prev_cam_tilt   = -1   # force initial tilt command on first loop tick
    lights          = LIGHT_MIN
    prev_lights     = -1   # force initial lights command on first loop tick
    last_hb         = 0.0
    prev_rt_pressed = False
    prev_alerts: list = []

    # track button press edges (only fire on press, not hold)
    prev_btns: dict[int, bool] = {}
    prev_hat = (0, 0)

    def just_pressed(btn: int) -> bool:
        cur = joy.get_button(btn)
        was = prev_btns.get(btn, False)
        return cur and not was

    def hat_just_pressed(x: int, y: int) -> bool:
        cur = joy.get_hat(0)
        return cur == (x, y) and prev_hat != (x, y)

    try:
        while True:
            pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt

            # ── button edges ───────────────────────────────────────────────────
            if just_pressed(BTN_A) and not collector.recording:
                if collector.start():
                    print(f"Recording started — episode {collector.total_episodes}")

            if just_pressed(BTN_X) and collector.recording:
                result = collector.finish(success=True)
                if result:
                    ep, n, _ = result
                    print(f"Episode {ep} saved  ({n} frames)  ✓ success")

            if just_pressed(BTN_B) and collector.recording:
                result = collector.finish(success=False)
                if result:
                    ep, n, _ = result
                    print(f"Episode {ep} saved  ({n} frames)  ✗ failure")

            if just_pressed(BTN_START):
                if set_arm(True):
                    armed = True
                    depth_setpoint = telem["depth"]
                    depth_pid.reset()
                    print(f"Armed  (depth setpoint → {depth_setpoint:.2f} m)")

            if just_pressed(BTN_BACK):
                if set_arm(False):
                    armed = False
                    print("Disarmed")


            # snapshot states for next tick's edge detection
            prev_btns = {b: joy.get_button(b)
                         for b in [BTN_A, BTN_B, BTN_X, BTN_Y,
                                   BTN_LB, BTN_RB, BTN_START, BTN_BACK]}
            prev_hat = joy.get_hat(0)

            # ── axes → manual control ──────────────────────────────────────────
            gain       = live_cfg["gain"]
            deadband   = live_cfg["deadband"]
            depth_rate = live_cfg["depth_rate"]
            depth_min  = live_cfg["depth_min"]
            depth_max  = live_cfg["depth_max"]

            # fwd     = apply_deadband(-joy.get_axis(AX_LEFT_Y),  deadband)
            # lat     = apply_deadband( joy.get_axis(AX_LEFT_X),  deadband)
            # yaw     = apply_deadband( joy.get_axis(AX_RIGHT_X), deadband)
            # dive    = apply_deadband( joy.get_axis(AX_RIGHT_Y), deadband)
            fwd     = -joy.get_axis(AX_LEFT_Y)
            lat     =  joy.get_axis(AX_LEFT_X)
            yaw     =  joy.get_axis(AX_RIGHT_X)
            # dive    =  apply_deadband( joy.get_axis(AX_RIGHT_Y), deadband)
            dive = 0.0

            x = int(fwd * 1000 * gain)
            y = int(lat * 1000 * gain)
            r = int(yaw * 1000 * gain * ROT_GAIN) 

            # depth setpoint update: stick deflection shifts target at depth_rate m/s
            # depth_setpoint += dive * depth_rate / SEND_HZ * DEPTH_GAIN
            # depth_setpoint = max(depth_min, min(depth_max, depth_setpoint))

            # PID controller: error -> vertical thruster command (500 = hover)
            depth_error = depth_setpoint - telem["depth"]
            z = depth_pid.update(depth_error, 1.0 / SEND_HZ,
                                 live_cfg["kp_depth"], live_cfg["ki_depth"], live_cfg["kd_depth"])

            # ── trigger gripper (RT: press = close, release = open) ───────────
            rt_pressed = joy.get_axis(AX_RT) > TRIGGER_THRESHOLD

            if rt_pressed and not prev_rt_pressed:
                set_gripper(False)
                print("Gripper: close")
            elif not rt_pressed and prev_rt_pressed:
                set_gripper(True)
                print("Gripper: open")

            prev_rt_pressed = rt_pressed

            # ── d-pad: camera tilt (up/down), lights (left/right) — hold to change
            hat = joy.get_hat(0)
            if hat[1] == 1:
                cam_tilt = max(1100, cam_tilt - CAM_TILT_STEP)
            elif hat[1] == -1:
                cam_tilt = min(1900, cam_tilt + CAM_TILT_STEP)
            if hat[0] == -1:
                lights = max(LIGHT_MIN, lights - LIGHT_STEP)
            elif hat[0] == 1:
                lights = min(LIGHT_MAX, lights + LIGHT_STEP)

            # ── bumpers: depth setpoint nudge ──────────────────────────────────
            if joy.get_button(BTN_RB):
                dive = -(depth_rate / SEND_HZ * DEPTH_GAIN)
                depth_setpoint -= depth_rate / SEND_HZ * DEPTH_GAIN
            if joy.get_button(BTN_LB):
                dive = -(depth_rate / SEND_HZ * DEPTH_GAIN)
                depth_setpoint += depth_rate / SEND_HZ * DEPTH_GAIN
            depth_setpoint = max(depth_min, min(depth_max, depth_setpoint))

            # ── record frame ───────────────────────────────────────────────────
            collector.add_frame(cam["frame"], telem, x, y, z, r, cam_tilt, rt_pressed,
                                dive=dive, depth_setpoint=depth_setpoint)

            # ── health / power alerts ──────────────────────────────────────────
            alerts = compute_alerts(telem)
            if alerts and alerts != prev_alerts:
                print(f"[ALERT] {'  |  '.join(alerts)}")
            prev_alerts = alerts

            # ── send ───────────────────────────────────────────────────────────
            now = time.monotonic()
            if now - last_hb >= 1.0:
                send_heartbeat()
                last_hb = now

            if armed:
                send_manual_control(x, y, z, r)

            if cam_tilt != prev_cam_tilt:
                set_camera_tilt(cam_tilt)
                prev_cam_tilt = cam_tilt

            if lights != prev_lights:
                set_lights(lights)
                prev_lights = lights

            # ── render ─────────────────────────────────────────────────────────
            if ENABLE_CAMERA:
                frame = cam["frame"]
                screen.blit(
                    pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
                    if frame is not None else placeholder,
                    (0, 0)
                )
            draw_hud(screen, font, telem, armed, x, y, z, r, cam_tilt,
                     depth_setpoint, lights, collector.recording,
                     collector.total_episodes, collector.total_successes)
            pygame.display.flip()
            clock.tick(SEND_HZ)

    except KeyboardInterrupt:
        pass
    finally:
        if collector.recording:
            collector.finish(success=False)
            print("Recording discarded on exit.")
        stop.set()
        if armed:
            set_arm(False)
        send_manual_control(0, 0, 500, 0)
        _set_servo_function(11, 184.0)
        _set_servo_function(13, 0.0)
        _set_servo_function(14, 0.0)
        _set_servo_function(16, 7.0)
        pygame.quit()
        print("Exited cleanly.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BlueROV2 Xbox teleop with LeRobot v3 data collection")
    ap.add_argument("task_name", help='Task label, e.g. "grab_coral". Data saved to D:\\BLUEROV2 data\\creo_pool\\<task_name>\\')
    args = ap.parse_args()
    main(args.task_name)
