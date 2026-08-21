"""
Xbox controller teleop for BlueROV2.

Control layout:
  Left  stick Y    forward / back
  Left  stick X    strafe left / right
  Right stick X    yaw
  Right stick Y    ascend / descend
  LT (axis 4)      gripper open
  RT (axis 5)      gripper close
  LB (button 4)    camera tilt down
  RB (button 5)    camera tilt up
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
from pathlib import Path

import av
import cv2
import numpy as np
import requests
import pygame

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BLUEROV_URL, DATA_ROOT

# ── config ─────────────────────────────────────────────────────────────────────

SEND_HZ   = 50
GAIN              = 0.5   # scale joystick → command (1.0 = full range, 0.5 = half)
DEADBAND          = 0.05
TRIGGER_THRESHOLD = 0.5   # trigger axis value to register as a press
CAM_TILT_STEP = 15    # PWM per tick while bumper held
CAM_W, CAM_H = 854, 480
HUD_H         = 160
WIN_W, WIN_H  = CAM_W, CAM_H + HUD_H

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
            p2 = _session.get(
                f"{MAV}/mavlink/vehicles/1/components/1/messages/SCALED_PRESSURE2",
                timeout=0.5).json()["message"]
            data["depth"] = (p2.get("press_abs", 1013.25) - 1013.25) / 98.0
        except Exception:
            pass
        time.sleep(0.2)


def camera_loop(shared: dict, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            container = av.open(SDP,
                                options={"protocol_whitelist": "file,rtp,udp,crypto"},
                                timeout=5)
            for packet in container.demux(video=0):
                if stop.is_set():
                    break
                for frame in packet.decode():
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.resize(img, (CAM_W, CAM_H))
                    shared["frame"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            container.close()
        except Exception as e:
            shared["cam_error"] = str(e)
            time.sleep(1)


# ── helpers ────────────────────────────────────────────────────────────────────

def apply_deadband(value: float, db: float = DEADBAND) -> float:
    return 0.0 if abs(value) < db else value


# ── data collection ────────────────────────────────────────────────────────────

class DataCollector:
    """Writes a LeRobot v2 dataset (parquet + H264 MP4) to disk at SEND_HZ.

    Layout:
      <root>/
        meta/info.json  episodes.jsonl  tasks.jsonl
        data/chunk-000/episode_XXXXXX.parquet
        videos/chunk-000/observation.images.camera/episode_XXXXXX.mp4

    Actions are normalised to [-1, 1]:  x/1000, y/1000, (z-500)/500, r/1000.
    State is [roll, pitch, yaw, depth] in radians / metres.
    """

    CHUNKS_SIZE = 1000

    def __init__(self, task_name: str, root: Path) -> None:
        self.task_name = task_name
        self.root      = root
        (root / "meta").mkdir(parents=True, exist_ok=True)

        info_path = root / "meta" / "info.json"
        if info_path.exists():
            self.info           = json.loads(info_path.read_text())
            self.total_episodes = self.info["total_episodes"]
            self.total_frames   = self.info["total_frames"]
        else:
            self.total_episodes = 0
            self.total_frames   = 0
            self.info           = self._build_info()
            self._flush_info()
            (root / "meta" / "tasks.jsonl").write_text(
                json.dumps({"task_index": 0, "task": task_name}) + "\n"
            )

        self._recording    = False
        self._rows: list   = []
        self._container    = None
        self._stream       = None
        self._vpath: Path  = Path()
        self._ep_start     = 0.0
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
        vdir  = (self.root / "videos" / f"chunk-{chunk:03d}"
                 / "observation.images.camera")
        vdir.mkdir(parents=True, exist_ok=True)
        self._vpath     = vdir / f"episode_{ep:06d}.mp4"
        self._container = av.open(str(self._vpath), mode="w")
        self._stream    = self._container.add_stream("libx264", rate=SEND_HZ)
        self._stream.width   = CAM_W
        self._stream.height  = CAM_H
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"preset": "fast", "crf": "23"}

        self._rows         = []
        self._ep_start     = time.monotonic()
        self._ep_frame_idx = 0
        self._recording    = True
        return True

    def add_frame(self, frame, telem: dict, x: int, y: int,
                  z: int, r: int) -> None:
        """Encode one video frame and append one data row. No-op if not recording."""
        if not self._recording:
            return
        img = (frame if frame is not None
               else np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8))
        av_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for pkt in self._stream.encode(av_frame.reformat(format="yuv420p")):
            self._container.mux(pkt)

        self._rows.append({
            "obs": np.array([telem["roll"], telem["pitch"],
                             telem["yaw"],  telem["depth"]], dtype=np.float32),
            "act": np.array([x / 1000.0, y / 1000.0,
                             (z - 500) / 500.0, r / 1000.0], dtype=np.float32),
            "ts":  time.monotonic() - self._ep_start,
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
        self.total_frames   += n
        self.total_episodes += 1

        with open(self.root / "meta" / "episodes.jsonl", "a") as f:
            f.write(json.dumps({
                "episode_index": ep,
                "tasks":   [self.task_name],
                "length":  n,
                "success": success,
            }) + "\n")
        self._flush_info()
        return ep, n, success

    # ── private ────────────────────────────────────────────────────────────────

    def _build_info(self) -> dict:
        return {
            "codebase_version": "v2.0",
            "robot_type":       "bluerov2",
            "total_episodes":   0,
            "total_frames":     0,
            "total_tasks":      1,
            "total_videos":     0,
            "total_chunks":     1,
            "chunks_size":      self.CHUNKS_SIZE,
            "fps":              SEND_HZ,
            "splits":           {"train": "0:0"},
            "data_path":  "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
            "video_path": ("videos/chunk-{chunk_index:03d}/{video_key}"
                           "/episode_{episode_index:06d}.mp4"),
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
                    "dtype": "float32", "shape": [4],
                    "names": ["roll", "pitch", "yaw", "depth"],
                },
                "action": {
                    "dtype": "float32", "shape": [4],
                    "names": ["x", "y", "z", "r"],
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
        n_chunks = max(1, (self.total_episodes + self.CHUNKS_SIZE - 1) // self.CHUNKS_SIZE)
        self.info.update({
            "total_episodes": self.total_episodes,
            "total_frames":   self.total_frames,
            "total_videos":   self.total_episodes,
            "total_chunks":   n_chunks,
            "splits":         {"train": f"0:{self.total_episodes}"},
        })
        (self.root / "meta" / "info.json").write_text(json.dumps(self.info, indent=2))

    def _write_parquet(self, path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        n    = len(self._rows)
        done = [False] * n
        done[-1] = True
        table = pa.table({
            "observation.state": pa.array([r["obs"].tolist() for r in self._rows],
                                          pa.list_(pa.float32())),
            "action":            pa.array([r["act"].tolist() for r in self._rows],
                                          pa.list_(pa.float32())),
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
             recording: bool = False, ep_count: int = 0):
    hud = pygame.Surface((WIN_W, HUD_H))
    hud.fill((15, 15, 15))
    arm_col = (80, 220, 80) if armed else (220, 80, 80)
    rec_col = (220, 60, 60) if recording else (120, 120, 120)
    rec_str = f"● REC ep{ep_count}" if recording else f"○ IDLE  {ep_count} saved"
    lines = [
        (f"{'● ARMED' if armed else '○ DISARMED':12s}   "
         f"Roll:{telem['roll']:+.2f}  Pitch:{telem['pitch']:+.2f}  "
         f"Yaw:{telem['yaw']:+.2f}  Depth:{telem['depth']:.2f}m", arm_col),
        (f"{rec_str:22s}  fwd:{x:+d}  lat:{y:+d}  thrtl:{z}  yaw:{r:+d}  cam:{cam_tilt}",
         rec_col),
        ("Start=arm  Back=disarm  A=rec-start  X=success  B=fail  LT=grip-open  RT=grip-close",
         (100, 100, 100)),
        ("Left-stick=move  Right-X=yaw  Right-Y=depth  LB/RB=cam-tilt",
         (100, 100, 100)),
    ]
    for i, (text, col) in enumerate(lines):
        hud.blit(font.render(text, True, col), (12, 10 + i * 34))
    screen.blit(hud, (0, CAM_H))


# ── main ───────────────────────────────────────────────────────────────────────

def main(task_name: str) -> None:
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

    # take servo ownership
    _set_servo_function(11, 0.0)
    _set_servo_function(16, 0.0)
    print("Servos ready")

    # motor direction — motors 1, 2, 5 are physically reversed on this vehicle
    for m in (1, 2, 5):
        _set_motor_direction(m, -1)
    print("Motor directions set (1, 2, 5 reversed)")

    data_root = DATA_ROOT / task_name
    collector = DataCollector(task_name, data_root)
    print(f"Data collector ready → {data_root}  ({collector.total_episodes} episodes already saved)")

    telem = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "depth": 0.0}
    cam   = {"frame": None, "cam_error": None}
    stop  = threading.Event()
    threads = [
        threading.Thread(target=telemetry_loop, args=(telem, stop), daemon=True),
        threading.Thread(target=camera_loop,    args=(cam,   stop), daemon=True),
        threading.Thread(target=_sender_loop,   args=(stop,),       daemon=True),
    ]
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
    cam_tilt        = 1500
    prev_cam_tilt   = cam_tilt
    last_hb         = 0.0
    prev_lt_pressed = False
    prev_rt_pressed = False

    # track button press edges (only fire on press, not hold)
    prev_btns: dict[int, bool] = {}

    def just_pressed(btn: int) -> bool:
        cur = joy.get_button(btn)
        was = prev_btns.get(btn, False)
        return cur and not was

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
                    print("Armed")

            if just_pressed(BTN_BACK):
                if set_arm(False):
                    armed = False
                    print("Disarmed")

            # snapshot button states for next tick's edge detection
            prev_btns = {b: joy.get_button(b)
                         for b in [BTN_A, BTN_B, BTN_X, BTN_Y,
                                   BTN_LB, BTN_RB, BTN_START, BTN_BACK]}

            # ── axes → manual control ──────────────────────────────────────────
            fwd     = apply_deadband(-joy.get_axis(AX_LEFT_Y))
            lat     = apply_deadband( joy.get_axis(AX_LEFT_X))
            yaw     = apply_deadband( joy.get_axis(AX_RIGHT_X))
            dive    = apply_deadband( joy.get_axis(AX_RIGHT_Y))

            x = int(fwd  * 1000 * GAIN)
            y = int(lat  * 1000 * GAIN)
            r = int(yaw  * 1000 * GAIN)
            z = int(500  - dive * 500 * GAIN)   # up = ascend (z > 500), down = descend

            # ── trigger gripper (edge on press) ────────────────────────────────
            lt_pressed = joy.get_axis(AX_LT) > TRIGGER_THRESHOLD
            rt_pressed = joy.get_axis(AX_RT) > TRIGGER_THRESHOLD

            if lt_pressed and not prev_lt_pressed:
                set_gripper(True)
                print("Gripper: open")
            if rt_pressed and not prev_rt_pressed:
                set_gripper(False)
                print("Gripper: close")

            prev_lt_pressed = lt_pressed
            prev_rt_pressed = rt_pressed

            # ── record frame ───────────────────────────────────────────────────
            collector.add_frame(cam["frame"], telem, x, y, z, r)

            # ── camera tilt ────────────────────────────────────────────────────
            if joy.get_button(BTN_LB):
                cam_tilt = max(1100, cam_tilt - CAM_TILT_STEP)
            if joy.get_button(BTN_RB):
                cam_tilt = min(1900, cam_tilt + CAM_TILT_STEP)

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

            # ── render ─────────────────────────────────────────────────────────
            frame = cam["frame"]
            screen.blit(
                pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
                if frame is not None else placeholder,
                (0, 0)
            )
            draw_hud(screen, font, telem, armed, x, y, z, r, cam_tilt,
                     collector.recording, collector.total_episodes)
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
        _set_servo_function(16, 7.0)
        pygame.quit()
        print("Exited cleanly.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BlueROV2 Xbox teleop with LeRobot v2 data collection")
    ap.add_argument("task_name", help='Task label, e.g. "grab_coral". Data saved to DATA_ROOT/<task_name>/ (see config.py)')
    args = ap.parse_args()
    main(args.task_name)
