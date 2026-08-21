"""
Keyboard teleoperation for BlueROV2 via MAVLink2Rest (BlueOS).
Shows live camera feed with HUD overlay in a single pygame window.

Controls:
  W / S       forward / backward
  A / D       strafe left / right
  Up / Down   ascend / descend
  Q / E       yaw left / right
  Space       arm / disarm toggle
  Esc         quit (disarms first)

Run from project root:
  python src/teleop.py
"""

import sys
import time
import threading
from pathlib import Path

import queue

import av
import cv2
import numpy as np
import requests
import pygame

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BLUEROV_URL

# ── config ─────────────────────────────────────────────────────────────────────

SEND_HZ   = 50
STEP      = 500
CAM_W     = 854     # display width  (scales 1920→854 keeping 16:9)
CAM_H     = 480     # display height
HUD_H     = 150     # HUD bar below camera
WIN_W, WIN_H = CAM_W, CAM_H + HUD_H

MAV    = f"{BLUEROV_URL}/mavlink2rest"
HEADER = {"system_id": 255, "component_id": 0, "sequence": 0}
SDP    = str(Path(__file__).parent.parent / "stream.sdp")


# ── MAVLink2Rest helpers ───────────────────────────────────────────────────────

# Shared session for keep-alive (reuses TCP connection, avoids per-request handshake)
_session = requests.Session()

# maxsize=1: main loop always overwrites with the latest command, never queues up lag
_cmd_queue: queue.Queue = queue.Queue(maxsize=1)


def _sender_loop(stop: threading.Event) -> None:
    """Background thread: drains _cmd_queue and POSTs to MAVLink2Rest."""
    while not stop.is_set():
        try:
            message = _cmd_queue.get(timeout=0.05)
            _session.post(
                f"{MAV}/mavlink",
                json={"header": HEADER, "message": message},
                timeout=0.1,
            )
        except (queue.Empty, Exception):
            pass


def _enqueue(message: dict) -> None:
    """Non-blocking put — drops the previous command if the queue is full."""
    try:
        _cmd_queue.put_nowait(message)
    except queue.Full:
        try:
            _cmd_queue.get_nowait()
        except queue.Empty:
            pass
        _cmd_queue.put_nowait(message)


def _post_sync(message: dict) -> bool:
    """Synchronous POST for arm/disarm/heartbeat where we need the result."""
    try:
        r = _session.post(f"{MAV}/mavlink", json={"header": HEADER, "message": message}, timeout=0.5)
        return r.ok
    except Exception:
        return False


def get_message(name: str) -> dict | None:
    try:
        r = _session.get(f"{MAV}/mavlink/vehicles/1/components/1/messages/{name}", timeout=0.5)
        return r.json()["message"] if r.ok else None
    except Exception:
        return None


def send_heartbeat() -> None:
    _post_sync({
        "type": "HEARTBEAT",
        "custom_mode": 0,
        "mavtype": {"type": "MAV_TYPE_GCS"},
        "autopilot": {"type": "MAV_AUTOPILOT_INVALID"},
        "base_mode": {"bits": 0},
        "system_status": {"type": "MAV_STATE_ACTIVE"},
        "mavlink_version": 3,
    })


def set_arm(armed: bool) -> bool:
    return _post_sync({
        "type": "COMMAND_LONG",
        "target_system": 1, "target_component": 1,
        "command": {"type": "MAV_CMD_COMPONENT_ARM_DISARM"},
        "confirmation": 0,
        "param1": 1.0 if armed else 0.0,
        "param2": 21196.0,
        "param3": 0.0, "param4": 0.0,
        "param5": 0.0, "param6": 0.0, "param7": 0.0,
    })


def send_manual_control(x: int, y: int, z: int, r: int) -> None:
    """Non-blocking: enqueues command for the sender thread."""
    _enqueue({
        "type": "MANUAL_CONTROL",
        "target": 1,
        "x": x, "y": y, "z": z, "r": r,
        "buttons": 0,
    })


def _param_id_chars(name: str) -> list:
    return list(name.ljust(16, "\x00")[:16])


def _set_servo_function(servo: int, value: float) -> None:
    _post_sync({"type": "PARAM_SET", "target_system": 1, "target_component": 1,
                "param_id": _param_id_chars(f"SERVO{servo}_FUNCTION"),
                "param_value": value,
                "param_type": {"type": "MAV_PARAM_TYPE_INT8"}})


def _do_set_servo(channel: int, pwm: float) -> None:
    """Enqueue a DO_SET_SERVO — non-blocking, goes through sender thread."""
    _enqueue({"type": "COMMAND_LONG", "target_system": 1, "target_component": 1,
              "command": {"type": "MAV_CMD_DO_SET_SERVO"}, "confirmation": 0,
              "param1": float(channel), "param2": pwm,
              "param3": 0.0, "param4": 0.0, "param5": 0.0, "param6": 0.0, "param7": 0.0})


def set_gripper(open: bool) -> None:
    """Open or close the gripper. SERVO11_FUNCTION must already be 0."""
    _do_set_servo(11, 1900.0 if open else 1100.0)


def set_camera_tilt(pwm: int) -> None:
    """Set camera tilt PWM (1100=down, 1500=center, 1900=up). SERVO16_FUNCTION must be 0."""
    _do_set_servo(16, float(pwm))


# ── background threads ─────────────────────────────────────────────────────────

def telemetry_loop(data: dict, stop: threading.Event) -> None:
    while not stop.is_set():
        att = get_message("ATTITUDE")
        if att:
            data["roll"]  = att.get("roll", 0)
            data["pitch"] = att.get("pitch", 0)
            data["yaw"]   = att.get("yaw", 0)
        p2 = get_message("SCALED_PRESSURE2")
        if p2:
            data["depth"] = (p2.get("press_abs", 1013.25) - 1013.25) / 98.0
        time.sleep(0.2)


def camera_loop(shared: dict, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            container = av.open(
                SDP,
                options={"protocol_whitelist": "file,rtp,udp,crypto"},
                timeout=5,
            )
            for packet in container.demux(video=0):
                if stop.is_set():
                    break
                for frame in packet.decode():
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.resize(img, (CAM_W, CAM_H))
                    # convert BGR → RGB for pygame
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    shared["frame"] = img
            container.close()
        except Exception as e:
            shared["cam_error"] = str(e)
            time.sleep(1)  # retry after brief pause


# ── HUD drawing ────────────────────────────────────────────────────────────────

def draw_hud(screen: pygame.Surface, font: pygame.font.Font,
             telem: dict, armed: bool,
             x: int, y: int, z: int, r: int, cam_tilt: int = 1500) -> None:
    hud = pygame.Surface((WIN_W, HUD_H))
    hud.fill((15, 15, 15))

    arm_col = (80, 220, 80) if armed else (220, 80, 80)
    lines = [
        (f"{'● ARMED' if armed else '○ DISARMED':12s}   "
         f"Roll:{telem['roll']:+.2f}  Pitch:{telem['pitch']:+.2f}  "
         f"Yaw:{telem['yaw']:+.2f}  Depth:{telem['depth']:.2f}m", arm_col),
        (f"Cmd  fwd:{x:+d}  lat:{y:+d}  thrtl:{z}  yaw:{r:+d}  cam:{cam_tilt}", (160, 160, 160)),
        ("WASD=move  ↑↓=depth  QE=yaw  [/]=cam tilt  G/F=gripper  Space=arm  Esc=quit", (100, 100, 100)),
    ]
    for i, (text, col) in enumerate(lines):
        hud.blit(font.render(text, True, col), (12, 12 + i * 22))

    screen.blit(hud, (0, CAM_H))


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # connectivity check
    try:
        r = requests.get(f"{MAV}/mavlink/vehicles/1/components/1/messages/HEARTBEAT", timeout=3)
        assert r.ok
        print("Connected to BlueROV2")
    except Exception:
        print(f"Cannot reach {MAV}")
        return

    # Take direct ownership of gripper and camera servos for the session
    _set_servo_function(11, 0.0)   # gripper  (was k_actuator1 = 184)
    _set_servo_function(16, 0.0)   # cam tilt (was RCPassThru7 = 7)
    print("Gripper ready (G=open  F=close)  Camera tilt ready ([ / ])")

    telem  = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "depth": 0.0}
    cam    = {"frame": None, "cam_error": None}
    stop   = threading.Event()

    threads = [
        threading.Thread(target=telemetry_loop, args=(telem, stop), daemon=True),
        threading.Thread(target=camera_loop,    args=(cam, stop),   daemon=True),
        threading.Thread(target=_sender_loop,   args=(stop,),       daemon=True),
    ]
    for t in threads:
        t.start()

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("BlueROV2 Teleop")
    font  = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    # placeholder shown before first camera frame
    placeholder = pygame.Surface((CAM_W, CAM_H))
    placeholder.fill((30, 30, 30))
    no_cam_text = font.render("Waiting for camera…", True, (120, 120, 120))
    placeholder.blit(no_cam_text, (CAM_W // 2 - 80, CAM_H // 2))

    armed         = False
    last_hb       = 0.0
    cam_tilt      = 1500   # 1100=down  1500=center  1900=up
    prev_cam_tilt = cam_tilt

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
                    if event.key == pygame.K_SPACE:
                        target = not armed
                        if set_arm(target):
                            armed = target
                            print("Armed" if armed else "Disarmed")
                        else:
                            print("Arm command rejected")
                    if event.key == pygame.K_g:
                        set_gripper(True)
                        print("Gripper: open")
                    if event.key == pygame.K_f:
                        set_gripper(False)
                        print("Gripper: close")

            # GCS heartbeat at 1 Hz
            now = time.monotonic()
            if now - last_hb >= 1.0:
                send_heartbeat()
                last_hb = now

            keys = pygame.key.get_pressed()
            x = y = r = 0
            z = 500

            if keys[pygame.K_w]:    x = +STEP
            if keys[pygame.K_s]:    x = -STEP
            if keys[pygame.K_a]:    y = -STEP
            if keys[pygame.K_d]:    y = +STEP
            if keys[pygame.K_UP]:   z = 500 + STEP
            if keys[pygame.K_DOWN]: z = 500 - STEP
            if keys[pygame.K_q]:    r = -STEP
            if keys[pygame.K_e]:    r = +STEP

            if keys[pygame.K_LEFTBRACKET]:
                cam_tilt = max(1100, cam_tilt - 25)
            if keys[pygame.K_RIGHTBRACKET]:
                cam_tilt = min(1900, cam_tilt + 25)

            if armed:
                send_manual_control(x, y, z, r)

            if cam_tilt != prev_cam_tilt:
                set_camera_tilt(cam_tilt)
                prev_cam_tilt = cam_tilt

            # draw camera frame
            frame = cam["frame"]
            if frame is not None:
                surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
                screen.blit(surf, (0, 0))
            else:
                screen.blit(placeholder, (0, 0))

            draw_hud(screen, font, telem, armed, x, y, z, r, cam_tilt)
            pygame.display.flip()
            clock.tick(SEND_HZ)

    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if armed:
            set_arm(False)
        send_manual_control(0, 0, 500, 0)
        _set_servo_function(11, 184.0)  # restore Actuator1
        _set_servo_function(16, 7.0)   # restore RCPassThru7
        pygame.quit()
        print("Exited cleanly.")


if __name__ == "__main__":
    main()
