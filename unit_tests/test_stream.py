"""
Stream test: pipe frames from system FFmpeg into a pygame window.

Run:  conda run -n bluerov python unit_tests/test_stream.py
      (ROV must be streaming on UDP port 5600)

Press Q or Escape to quit.
"""

import subprocess
import threading
import time
import sys
import numpy as np
import pygame
from pathlib import Path

CAM_W, CAM_H = 854, 480
FFMPEG = "ffmpeg"
SDP    = str(Path(__file__).parent.parent / "stream.sdp")

# ── frame reader ──────────────────────────────────────────────────────────────

shared = {"frame": None, "error": None, "fps": 0.0}


def _reader():
    cmd = [
        FFMPEG,
        "-loglevel",  "warning",
        "-protocol_whitelist", "file,udp,rtp,crypto",
        "-i",          SDP,
        "-f",          "rawvideo",
        "-pix_fmt",    "rgb24",
        "-vf",         f"scale={CAM_W}:{CAM_H}",
        "pipe:1",
    ]
    print("Starting FFmpeg:", " ".join(cmd))
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        shared["error"] = f"ffmpeg not found at {FFMPEG}"
        return

    frame_bytes = CAM_W * CAM_H * 3
    count = 0
    t0 = time.monotonic()

    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            err = proc.stderr.read().decode(errors="replace")
            shared["error"] = f"FFmpeg stream ended. stderr:\n{err}"
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((CAM_H, CAM_W, 3))
        shared["frame"] = frame
        count += 1
        elapsed = time.monotonic() - t0
        if elapsed >= 1.0:
            shared["fps"] = count / elapsed
            count = 0
            t0 = time.monotonic()

    proc.terminate()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((CAM_W, CAM_H))
    pygame.display.set_caption("Stream test — Q to quit")
    font  = pygame.font.SysFont("monospace", 18)
    clock = pygame.time.Clock()

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    placeholder = pygame.Surface((CAM_W, CAM_H))
    placeholder.fill((20, 20, 20))

    start = time.monotonic()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                pygame.quit(); sys.exit()

        frame = shared["frame"]
        err   = shared["error"]

        if err:
            screen.fill((30, 0, 0))
            for i, line in enumerate(err.splitlines()[:20]):
                screen.blit(font.render(line, True, (220, 80, 80)), (10, 10 + i * 22))
        elif frame is not None:
            surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            screen.blit(surf, (0, 0))
            fps_text = font.render(f"{shared['fps']:.1f} fps", True, (0, 255, 0))
            screen.blit(fps_text, (10, 10))
        else:
            screen.blit(placeholder, (0, 0))
            wait = time.monotonic() - start
            screen.blit(font.render(f"Waiting for first frame… {wait:.1f}s", True, (150, 150, 150)),
                        (CAM_W // 2 - 160, CAM_H // 2))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
