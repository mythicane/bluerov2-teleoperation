# bluerov2-teleoperation

Teleoperation, LeRobot-format data collection, and ACT policy deployment for a
BlueROV2 (standard configuration) running BlueOS. Includes a small research
study on identifying and rejecting environmental (current) disturbances from
the onboard IMU signal.

This repo ships **no recorded episodes** — the `data/grab <color> rod[ under
currents]/` folders are empty skeletons (`data/`, `meta/`, `videos/`
subfolders) so you can either record your own data with
`src/controller_teleop.py`, or drop in the published dataset. The dataset
this project was developed against is on HuggingFace:

**https://huggingface.co/datasets/Mythicane/bluerov_pool_data**

## Hardware assumptions

- BlueROV2 (standard configuration, 6 vectored thrusters — not the 8-thruster
  Heavy config) running BlueOS, reachable over the Fathom-X tether at
  `192.168.2.2` (default — change in `config.py` if yours differs).
- Xbox controller (Series X/S layout; `src/controller_teleop.py` and
  `deployment_scripts/deploy_bluerov.py`) or keyboard (`src/teleop.py`).
- No GPS, DVL, or rangefinder on this vehicle — depth control is a PID loop
  driven off the external pressure sensor, not a full position estimate.

## Setup

Create the dedicated conda environment (Python 3.10, PyTorch, LeRobot, etc.):

```bash
conda env create -f environment.yml
conda activate bluerov
```

You'll also need `ffmpeg` on your `PATH` (used for the camera pipe in
`controller_teleop.py` / `deploy_bluerov.py`) and, on Windows, the BlueOS
video stream configured to RTP so `stream.sdp` matches your vehicle's IP.

Edit `config.py`:
- `BLUEROV_URL` — companion computer address (default `http://192.168.2.2`).
- `TLOG_DIR` — where QGroundControl saves `.tlog` files, if you use
  `src/explore.py`.
- `DATA_ROOT` — where task dataset folders live. Defaults to `REPO_ROOT/data`
  (i.e. the `data/grab <color> rod/` folders already in this repo).

## Teleoperation

Keyboard (no data recording, just flying + gripper/camera control):

```bash
python src/teleop.py
```
`W/S` forward/back, `A/D` strafe, `↑/↓` depth, `Q/E` yaw, `Space` arm/disarm,
`G/F` gripper, `[`/`]` camera tilt, `Esc` quit.

Xbox controller, with recording (this is also the data-collection entry
point):

```bash
python src/controller_teleop.py <task_name>
```
`task_name` becomes both the LeRobot task label and the output folder name
under `DATA_ROOT` (e.g. `"grab red rod"` → `data/grab red rod/`). Left stick = forward/strafe, right
stick X = yaw, `RT` = gripper, `LB`/`RB` = depth setpoint, D-pad =
camera/lights, `Start`/`Back` = arm/disarm, `A` = start episode, `X`/`B` =
save episode success/failure. A depth-hold PID (tunable live via
`utils/tune_pid.py`, backed by `pid_configs.yaml`) keeps depth stable so you
only drive laterally.

`src/old_controller_teleop.py` is kept for reference — an earlier version of
the controller script (LeRobot **v2** dataset format, simpler HUD/telemetry,
no depth-PID/lights/health-alerts). `src/controller_teleop.py` is the current,
actively used script; reach for the old one only if you specifically need v2
output.

## Data format

Each episode is written as a LeRobot v3 dataset (`data/chunk-*/episode_*.parquet`
+ `videos/observation.images.camera/chunk-*/episode_*.mp4` + `meta/`).

- **`observation.state`** (30-dim float32): `roll, pitch, yaw, depth,
  depth_setpoint, pid_z, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
  baro_press_hpa, enclosure_temp_c, thr1_pwm..thr6_pwm, rollspeed,
  pitchspeed, yawspeed, climb, vibration_x, vibration_y, vibration_z,
  ekf_compass_variance, ekf_pos_horiz_variance, ekf_pos_vert_variance`.
  `acc_*`/`gyro_*` are the raw onboard IMU (RAW_IMU); `roll/pitch/yaw` and
  `rollspeed/pitchspeed/yawspeed` are the EKF-fused attitude/rates
  (ATTITUDE) — these are deliberately kept separate. `baro_press_hpa`/
  `enclosure_temp_c` are the internal enclosure barometer, not the external
  depth sensor. See the docstring on `DataCollector` in
  `src/controller_teleop.py` for the full provenance of every field,
  including which fields were considered and dropped (no GPS/DVL/rangefinder
  on this vehicle, confirmed hard-zero via BlueOS).
- **`action`** (6-dim float32): `x, y, dive, r, cam_tilt, gripper`, all
  normalised to roughly `[-1, 1]`.
- Video: 854×480 h264/yuv420p @ 50 fps, no audio.

Validate the dataset format (schema, timestamp grid, stats.json, LeRobot
loader acceptance) with:

```bash
python test_dataset_format.py
```

Inspect a single episode's raw contents:

```bash
python inspect_parquet.py "data/grab red rod/data/chunk-000/episode_000000.parquet"
```

## Annotating episodes

A companion tool, [bluerov2-annotation-tool](https://github.com/mythicane/bluerov2-annotation-tool),
exists for reviewing recorded episodes in a browser: watch each episode's
video, mark it valid/invalid and success/fail, add notes, star ones worth
revisiting, and get a stats/trajectory dashboard. Point its `DATA_ROOT` at
this repo's `data/` folder (or wherever you've configured `DATA_ROOT` to
point) to use it.

## Model inference / DAgger

`deployment_scripts/deploy_bluerov.py` currently only supports **ACT**
checkpoints (it imports `ACTPolicy` directly and does ACT-specific checkpoint
config patching). It runs the policy on the vehicle with a temporal-ensembling
action chunker, plus DAgger-style human-override data collection (stick input
above a deflection threshold takes over from the policy and is logged with
`is_intervention=True`):

```bash
python deployment_scripts/deploy_bluerov.py \
    --checkpoint path/to/checkpoint.zip \
    --task-name grab-red-rod \
    --n-action-steps 25 \
    --ensemble-coeff 0.01
```

`Start`/`Back` arm/disarm, `A` starts a DAgger episode (policy drives), `X`/`B`
save success/failure, `Y` toggles a policy-only TEST mode with no recording.
DAgger episodes are written under `DATA_ROOT/dagger_data/<task-name>/` in the
same LeRobot v3 format described above, with one extra `is_intervention`
column. `--checkpoint` accepts either a `.zip` (auto-extracted) or an already
unzipped checkpoint directory containing `config.json`.

Train your own ACT checkpoint with [LeRobot](https://github.com/huggingface/lerobot)
against a `data/grab <color> rod[ under currents]` folder (or the HuggingFace
dataset above) as the `--dataset.root`.

[LeRobot](https://github.com/huggingface/lerobot) also supports training
other policy architectures (e.g. Diffusion Policy, SmolVLA, Pi0) on this same
dataset — `deploy_bluerov.py` just doesn't know how to run them yet. Swapping
in a different policy for inference would mean generalizing the checkpoint
loading (currently hardcoded to `ACTPolicy`) and, for language-conditioned
(VLA-style) policies, adding a task/instruction string to `make_obs()`, which
doesn't exist today.

## Perturbation-rejection study

`perturbation_rejection/` is a self-contained study asking: can we separate
the BlueROV2's own thruster-driven acceleration, Φ(u(t)), from periodic
environmental disturbances, Δ(t), using only onboard IMU + commanded PWM? See
`perturbation_rejection/method.md` for the full assumptions. In short: fit Φ
on the calm ("no currents") pooled dataset, then use a harmonic Kalman filter
to isolate Δ on the "under currents" dataset, and check that Φ stays
consistent across both once Δ is removed.

Run the full pipeline (fits, ablations, PSDs, and an HTML report) from
`perturbation_rejection/`:

```bash
cd perturbation_rejection
python run.py
```

This expects the `data/grab <color> rod[ under currents]` folders to be
populated (see Data format above) — it pools all 4 rod colors per condition. A
pre-generated example report/figures from one run are checked in under
`perturbation_rejection/outputs/` (`report.html`, `results.json`,
`heavy.json`, `preview/*.png`) so you can see the expected output without
running it yourself.

## Repo layout

```
config.py                     BlueROV2 URL, data root, tlog dir
environment.yml                conda env spec ("bluerov")
pid_configs.yaml               live-tunable depth PID + lateral gain/deadband
stream.sdp                     RTP camera stream descriptor for ffmpeg/PyAV

src/
  teleop.py                    keyboard teleop, no recording
  controller_teleop.py         Xbox teleop + LeRobot v3 data collection (DataCollector)
  old_controller_teleop.py     earlier Xbox teleop, no recording (legacy reference)
  explore.py                   interactive QGroundControl .tlog explorer
  test_gripper.py              gripper servo control scratch/debug script

deployment_scripts/
  deploy_bluerov.py            ACT policy deployment + DAgger collection (DAggerCollector)

utils/
  tune_pid.py                  Tk GUI for live-editing pid_configs.yaml

unit_tests/                    hardware-adjacent unit tests (av, ffmpeg pipe, depth PID, stream)
test_dataset_format.py         LeRobot v3 dataset schema/format compliance test
inspect_parquet.py             dump one episode parquet's schema + rows

perturbation_rejection/        Φ/Δ disturbance-rejection study (see above)

data/
  grab <color> rod[ under currents]/   empty dataset folder skeletons (data/, meta/, videos/)
```

## License

MIT — see `LICENSE`.
