"""
One-shot stream diagnostic: grab 3 frames via system FFmpeg and dump stderr.
Run: conda run -n bluerov python unit_tests/test_ffmpeg_pipe.py
"""
import subprocess
from pathlib import Path

FFMPEG = "ffmpeg"
SDP    = str(Path(__file__).parent.parent / "stream.sdp")
CAM_W, CAM_H = 854, 480

cmd = [
    FFMPEG,
    "-loglevel", "info",
    "-protocol_whitelist", "file,udp,rtp,crypto",
    "-i", SDP,
    "-f", "rawvideo",
    "-pix_fmt", "rgb24",
    "-vf", f"scale={CAM_W}:{CAM_H}",
    "-frames:v", "3",
    "pipe:1",
]
print("CMD:", " ".join(cmd))
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
try:
    stdout, stderr = proc.communicate(timeout=12)
except subprocess.TimeoutExpired:
    proc.kill()
    stdout, stderr = proc.communicate()
    print("RESULT: TIMEOUT — no frames in 12s")
else:
    frame_bytes = CAM_W * CAM_H * 3
    n_frames = len(stdout) // frame_bytes
    print(f"RESULT: got {len(stdout)} bytes = {n_frames} frames")

print("--- FFmpeg stderr ---")
print(stderr.decode(errors="replace")[-3000:])
